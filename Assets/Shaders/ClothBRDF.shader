// Credit to HeyCat-svg
// https://github.com/HeyCat-svg/GPU-Cloth-Simulation/blob/master/Assets/Scripts/ClothBRDF.shader

Shader "Custom/ClothBRDF"
{
    Properties {
        _MainTex ("Texture", 2D) = "white" {}
        _NormalMap ("Normal Map", 2D) = "bump" {}

        _Alpha ("Alpha", Float) = 0.5
        _K_amp ("K_amp", Float) = 0.5
        
        _Metallic ("Metallic", Float) = 0.5
    }

    SubShader {
        Tags {"RenderType"="Opaque"}

        Pass {
            Tags {"LightMode"="ForwardBase"}
            CULL BACK

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            #pragma multi_compile_fwbase

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "UnityStandardBRDF.cginc"

            struct v2f {
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float4 pos : SV_POSITION;

                half3 tspace0 : TEXCOORD2;
                half3 tspace1 : TEXCOORD3;
                half3 tspace2 : TEXCOORD4;
            };
            
            StructuredBuffer<float4> Positions;
            StructuredBuffer<float2> Texcoords;
            StructuredBuffer<int> Indices;
            StructuredBuffer<float4> Normals;
            StructuredBuffer<float4> Tangents;
            
            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _NormalMap;
            float4 _NormalMap_ST;
            float _Alpha;
            float _K_amp;
            float _Metallic;

            float3 Schlick_F(float3 F0, float LdotH) {
                float3 unit = float3(1.0, 1.0, 1.0);
                return F0 + (unit - F0) * Pow5(1.0 - LdotH);
            }

            float Cloth_D(float alpha, float k_amp, float NdotH) {
                float a2 = alpha * alpha;
                float d2 = NdotH * NdotH;
                float fact = UNITY_INV_PI * (1 - step(NdotH, 0)) / (1 + k_amp * a2);
                return fact * (1 + (k_amp * exp(d2 / (a2 * (d2 - 1)))) / pow(1 - d2, 2));
            }

            float3 My_BRDF(float NdotL, float NdotV, float VdotH, float NdotH, float3 LdotH, float3 albedo, float3 F0) {
                float3 F = Schlick_F(F0, LdotH);
                float D = Cloth_D(_Alpha, _K_amp, NdotH);
                float3 unit = float3(1.0, 1.0, 1.0);

                float3 diff = UNITY_INV_PI * (unit - F) * albedo;
                float3 spec = F * D / (4 * (NdotL + NdotV - NdotL * NdotV));
                
                return saturate(diff + spec);
            }

            v2f vert(uint id : SV_VertexID) {
                v2f o;
                int idx = Indices[id];
                float4 worldPos = Positions[idx];
                float3 wNormal = Normals[idx].xyz;
                o.pos = UnityWorldToClipPos(worldPos);
                o.uv = TRANSFORM_TEX(Texcoords[idx], _MainTex);
                o.worldPos = worldPos.xyz;

                float4 wTangent = Tangents[idx];
                half tangentSign = wTangent.w * unity_WorldTransformParams.w;
                half3 wBitangent = cross(wNormal, wTangent.xyz) * tangentSign;
                // tangent space to world space
                o.tspace0 = half3(wTangent.x, wBitangent.x, wNormal.x);
                o.tspace1 = half3(wTangent.y, wBitangent.y, wNormal.y);
                o.tspace2 = half3(wTangent.z, wBitangent.z, wNormal.z);

                return o;
            }

            half4 frag(v2f i) : SV_Target {
                float4 mainTex = tex2D(_MainTex, i.uv);
                float4 normalTex = tex2D(_NormalMap, i.uv);

                // Vectors
                float3 L = UnityWorldSpaceLightDir(i.worldPos);
                float3 V = normalize(_WorldSpaceLightPos0.xyz - i.worldPos.xyz);
                float3 H = Unity_SafeNormalize(L + V);

                float3 tnomral = UnpackNormal(normalTex);
                float3 N;
                N.x = dot(i.tspace0, tnomral);
                N.y = dot(i.tspace1, tnomral);
                N.z = dot(i.tspace2, tnomral);

                float NdotL = saturate(dot(N, L));
                float NdotH = saturate(dot(N, H));
                float NdotV = saturate(dot(N, V));
                float VdotH = saturate(dot(V, H));
                float LdotH = saturate(dot(L, H));

                half oneMinusReflectivity;
                half3 specColor;        // F0
                float3 albedo = DiffuseAndSpecularFromMetallic(mainTex.rgb, _Metallic, specColor, oneMinusReflectivity);

                float3 directLight = My_BRDF(NdotL, NdotV, VdotH, NdotH, LdotH, albedo, specColor) * NdotL;

                float4 col = float4(directLight * _LightColor0.rgb * 5, 1);
                col += float4(UNITY_LIGHTMODEL_AMBIENT.xyz * albedo * 5, 1);// + float4(0.1, 0.1, 0.1, 0);
                
                return col;
            }

            ENDCG
        }

        Pass {
            Tags {"LightMode"="ForwardBase"}
            CULL FRONT

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            #pragma multi_compile_fwbase

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "UnityStandardBRDF.cginc"

            struct v2f {
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float4 pos : SV_POSITION;

                half3 tspace0 : TEXCOORD2;
                half3 tspace1 : TEXCOORD3;
                half3 tspace2 : TEXCOORD4;
            };

            StructuredBuffer<int> Indices;
            StructuredBuffer<float4> Positions;
            StructuredBuffer<float2> Texcoords;
            StructuredBuffer<float4> Normals;
            StructuredBuffer<float4> Tangents;

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _NormalMap;
            float4 _NormalMap_ST;
            float _Alpha;
            float _K_amp;
            float _Metallic;


            float3 Schlick_F(float3 F0, float LdotH) {
                float3 unit = float3(1.0, 1.0, 1.0);
                return F0 + (unit - F0) * Pow5(1.0 - LdotH);
            }

            float Cloth_D(float alpha, float k_amp, float NdotH) {
                float a2 = alpha * alpha;
                float d2 = NdotH * NdotH;
                float fact = UNITY_INV_PI * (1 - step(NdotH, 0)) / (1 + k_amp * a2);
                return fact * (1 + (k_amp * exp(d2 / (a2 * (d2 - 1)))) / pow(1 - d2, 2));
            }

            float3 My_BRDF(float NdotL, float NdotV, float VdotH, float NdotH, float3 LdotH, float3 albedo, float3 F0) {
                float3 F = Schlick_F(F0, LdotH);
                float D = Cloth_D(_Alpha, _K_amp, NdotH);
                float3 unit = float3(1.0, 1.0, 1.0);

                float3 diff = UNITY_INV_PI * (unit - F) * albedo;
                float3 spec = F * D / (4 * (NdotL + NdotV - NdotL * NdotV));
                
                return saturate(diff + spec);
            }

            v2f vert(uint id : SV_VertexID) {
                v2f o;
                int idx = Indices[id];
                float4 worldPos = Positions[idx];
                float3 wNormal = Normals[idx].xyz;
                o.pos = UnityWorldToClipPos(worldPos);
                o.uv = TRANSFORM_TEX(Texcoords[idx], _MainTex);
                o.worldPos = worldPos.xyz;

                float4 wTangent = Tangents[idx];
                half tangentSign = wTangent.w * unity_WorldTransformParams.w;
                half3 wBitangent = cross(wNormal, wTangent.xyz) * tangentSign;
                // tangent space to world space
                o.tspace0 = half3(wTangent.x, wBitangent.x, wNormal.x);
                o.tspace1 = half3(wTangent.y, wBitangent.y, wNormal.y);
                o.tspace2 = half3(wTangent.z, wBitangent.z, wNormal.z);

                return o;
            }

            half4 frag(v2f i) : SV_Target {
                float4 mainTex = tex2D(_MainTex, i.uv);
                float4 normalTex = tex2D(_NormalMap, i.uv);

                // Vectors
                float3 L = UnityWorldSpaceLightDir(i.worldPos);
                float3 V = normalize(_WorldSpaceLightPos0.xyz - i.worldPos.xyz);
                float3 H = Unity_SafeNormalize(L + V);

                float3 tnomral = UnpackNormal(normalTex);
                float3 N;
                N.x = dot(i.tspace0, tnomral);
                N.y = dot(i.tspace1, tnomral);
                N.z = dot(i.tspace2, tnomral);
                
                N = -N;         // 背面反向

                float NdotL = saturate(dot(N, L));
                float NdotH = saturate(dot(N, H));
                float NdotV = saturate(dot(N, V));
                float VdotH = saturate(dot(V, H));
                float LdotH = saturate(dot(L, H));

                half oneMinusReflectivity;
                half3 specColor;        // F0
                float3 albedo = DiffuseAndSpecularFromMetallic(mainTex.rgb, _Metallic, specColor, oneMinusReflectivity);

                float3 directLight = My_BRDF(NdotL, NdotV, VdotH, NdotH, LdotH, albedo, specColor) * NdotL;

                float4 col = float4(directLight * _LightColor0.rgb * 5, 1);
                col += float4(UNITY_LIGHTMODEL_AMBIENT.xyz * albedo * 5, 1);// + float4(0.1, 0.1, 0.1, 0);

                return col;
            }

            ENDCG
        }
    }
}
