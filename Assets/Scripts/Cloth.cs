using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;

public class Cloth : MonoBehaviour
{
    [Header("Cloth Properties")] 
    public int rowVertexCount = 32;
    public int colVertexCount = 32;
    public float gridSize = 0.2f;
    public int distanceIterCount = 3;
    public int bendingIterCount = 1;
    public int collisionIterCount = 1;
    [Range(1, 10)] public int deltaTimeDivideCount = 6;
    [Range(0.0f, 1.0f)] public float damping = 1.0f;
    [Range(0.0f, Mathf.PI)] public float bendingAngle = Mathf.PI;
    public Vector4 gravity = new Vector4(0, -9.8f, 0, 0);

    public GameObject leftFixPoint, rightFixPoint;
    
    [Header("Compute Shaders")] 
    public ComputeShader pbdComputeShader;
    public ComputeShader normalComputeShader;
    public ComputeShader tangentComputeShader;

    [Header("Cloth Material")] 
    public Material material;

    private int vertexCount;
    private int indexCount;
    private Vector4 leftFixedPosModel, rightFixedPosModel;
    private float dt;
    private Vector3Int threadGroupCount;

    private Vector4[] positions;
    private Vector2[] texCoords;
    private Vector4[] normals;
    private Vector4[] tangents;
    private int[] indices;
    private Matrix4x4[] modelMatrices = new Matrix4x4[2];
    // private uint[] bitMap;

    private ComputeBuffer[] positionBufferPool;
    private ComputeBuffer texCoordBuffer;
    private ComputeBuffer indexBuffer;
    private ComputeBuffer normalBuffer;
    private ComputeBuffer tangentBuffer;
    // private ComputeBuffer bitMapBuffer;
    
    // PBD Kernels
    private int pbdUpdatePositionsKernel;
    private int pbdProjectDistanceConstraintsKernel;
    private int pbdProjectBendingConstraintsKernel;
    private int pbdProjectFixConstraintsKernel;
    private int pbdProjectSphereCollisionsConstraintsKernel;
    // Normal Compute Kernels
    private int normalComputeKernel;
    // Tangent Compute Kernels
    private int tangentComputeKernel;
    
    private bool hasInit = false;

    private GameObject[] Spheres;
    private string sphereTag = "Sphere";
    
    private void Start()
    {
        InitClothData();
        InitComputeShader();
        InitMaterial();
        InitCollisionObjects();
        hasInit = true;
    }

    private void InitClothData()
    {
        vertexCount = rowVertexCount * colVertexCount;
        indexCount = 6 * (rowVertexCount - 1) * (colVertexCount - 1);
        threadGroupCount = new Vector3Int(rowVertexCount / 8, colVertexCount / 8, 1);
        
        positions = new Vector4[vertexCount];
        texCoords = new Vector2[vertexCount];
        indices = new int[indexCount];
        normals = new Vector4[vertexCount];
        tangents = new Vector4[vertexCount];

        //bitMap = new uint[vertexCount];        

        positions.Initialize();
        texCoords.Initialize();
        indices.Initialize();
        normals.Initialize();
        tangents.Initialize();
        
        //leftFixedPosModel = new Vector4(0, 0, 0, 1);
        //rightFixedPosModel = new Vector4(gridSize * (colVertexCount - 1), 0, 0, 1);

        Transform transform1 = this.transform;
        Vector4 leftFixedPosWorld = leftFixPoint.transform.position;
        Vector4 rightWorld = transform1.right * gridSize;
        Vector4 upWorld = transform1.up * gridSize;
        Vector4 forwardWorld = transform1.forward * gridSize;
        //Vector4 rightObj = new Vector4(gridSize, 0, 0, 0);
        //Vector4 upObj = new Vector4(0, gridSize, 0, 0);
        
        int id = 0;
        for (var row = 0; row < rowVertexCount; row++)
        {
            for (var col = 0; col < colVertexCount; col++)
            {
                id = row * colVertexCount + col;
                positions[id] = leftFixedPosWorld + col * rightWorld - row * forwardWorld;
                texCoords[id] = new Vector2(row / (float)(rowVertexCount - 1),
                    col / (float)(colVertexCount - 1));
            }
        }

        id = 0;
        for (var row = 0; row < rowVertexCount - 1; row++)
        {
            for (var col = 0; col < colVertexCount - 1; col++)
            {
                var upLeft = row * colVertexCount + col;
                var upRight = row * colVertexCount + col + 1;
                var downLeft = (row + 1) * colVertexCount + col;
                var downRight = (row + 1) * colVertexCount + col + 1;

                indices[id] = upLeft;
                indices[id + 1] = downLeft;
                indices[id + 2] = downRight;
                indices[id + 3] = upLeft;
                indices[id + 4] = downRight;
                indices[id + 5] = upRight;
                id += 6;
            }
        }

        /*for (int i = 0; i < vertexCount; i++)
        {
            bitMap[i] = (uint)0x3f;
        }*/

        dt = Time.fixedDeltaTime / (float)deltaTimeDivideCount;

        UpdateModelMatrix();
    }

    private void InitComputeShader()
    {
        InitComputeBuffer();
        
        InitPBDComputeShader();
        InitNormalComputeShader();
        InitTangentComputeShader();
    }

    private void InitComputeBuffer()
    {
        //positionBuffer = new ComputeBuffer(vertexCount, 16);
        //positionBuffer.SetData(positions);

        positionBufferPool = new ComputeBuffer[3];
        for (int i = 0; i < 3; i++)
        {
            positionBufferPool[i] = new ComputeBuffer(vertexCount, 16);
            positionBufferPool[i].SetData(positions);
        }

        texCoordBuffer = new ComputeBuffer(vertexCount, 8);
        texCoordBuffer.SetData(texCoords);

        indexBuffer = new ComputeBuffer(indexCount, 4);
        indexBuffer.SetData(indices);

        normalBuffer = new ComputeBuffer(vertexCount, 16);
        normalBuffer.SetData(normals);
        
        tangentBuffer = new ComputeBuffer(vertexCount, 16);
        tangentBuffer.SetData(tangents);

        // Debug.Log("Fixed Vertex Count: " + fixedVertexCount);
        
        // fixedRowColBuffer = new ComputeBuffer(fixedVertexCount, 4);
        // fixedRowColBuffer.SetData(fixedRowCols);
        
        //bitMapBuffer = new ComputeBuffer(vertexCount, 4);
    }

    void InitPBDComputeShader()
    {
        pbdUpdatePositionsKernel = pbdComputeShader.FindKernel("UpdatePositions");
        pbdProjectDistanceConstraintsKernel = pbdComputeShader.FindKernel("ProjectDistanceConstraints");
        pbdProjectBendingConstraintsKernel = pbdComputeShader.FindKernel("ProjectBendingConstraints");
        pbdProjectFixConstraintsKernel = pbdComputeShader.FindKernel("ProjectFixConstraints");
        pbdProjectSphereCollisionsConstraintsKernel = pbdComputeShader.FindKernel("ProjectSphereCollisionsConstraints");
        
        pbdComputeShader.SetInt("colVertexCount", colVertexCount);
        pbdComputeShader.SetInt("rowVertexCount", rowVertexCount);
        pbdComputeShader.SetVector("gravityAcceleration", gravity);
        pbdComputeShader.SetFloat("damping", damping);
        pbdComputeShader.SetFloat("dt", dt);
        pbdComputeShader.SetFloat("gridSize", gridSize);
        pbdComputeShader.SetFloat("bendingAngle", bendingAngle);
        pbdComputeShader.SetMatrix("modelMatrix", modelMatrices[0]);
        pbdComputeShader.SetMatrix("modelMatrixInv", modelMatrices[1]);
    }

    void InitNormalComputeShader()
    {
        normalComputeKernel = normalComputeShader.FindKernel("NormalCompute");
        
        normalComputeShader.SetInt("colVertexCount", colVertexCount);
        normalComputeShader.SetInt("rowVertexCount", rowVertexCount);
        
        normalComputeShader.SetBuffer(normalComputeKernel, "Positions", positionBufferPool[1]);
        normalComputeShader.SetBuffer(normalComputeKernel, "Normals", normalBuffer);   
    }

    void InitTangentComputeShader()
    {
        tangentComputeKernel = tangentComputeShader.FindKernel("TangentCompute");
        
        tangentComputeShader.SetInt("colVertexCount", colVertexCount);
        tangentComputeShader.SetInt("rowVertexCount", rowVertexCount);
        
        tangentComputeShader.SetBuffer(tangentComputeKernel, "Positions", positionBufferPool[1]);
        tangentComputeShader.SetBuffer(tangentComputeKernel, "Normals", normalBuffer);
        tangentComputeShader.SetBuffer(tangentComputeKernel, "TexCoords", texCoordBuffer);
        tangentComputeShader.SetBuffer(tangentComputeKernel, "Tangents", tangentBuffer);
    }

    private void InitMaterial()
    {
        material.SetBuffer("Positions", positionBufferPool[1]);
        material.SetBuffer("Texcoords", texCoordBuffer);
        material.SetBuffer("Indices", indexBuffer);
        material.SetBuffer("Normals", normalBuffer);
        material.SetBuffer("Tangents", tangentBuffer);
        
    }

    private void InitCollisionObjects()
    {
        Spheres = GameObject.FindGameObjectsWithTag(sphereTag);
    }

    private void Update()
    {
        UpdateMaterial();
    }

    private void FixedUpdate()
    {
        if (!hasInit)
            return;

        UpdateModelMatrix();
        pbdComputeShader.SetMatrix("modelMatrix", modelMatrices[0]);
        pbdComputeShader.SetMatrix("modelMatrixInv", modelMatrices[1]);
        var lpos = leftFixPoint.transform.localPosition;
        var rpos = rightFixPoint.transform.localPosition;
        pbdComputeShader.SetVector("leftFixPos", new Vector4(lpos.x, lpos.y, lpos.z, 1));
        pbdComputeShader.SetVector("rightFixPos", new Vector4(rpos.x, rpos.y, rpos.z, 1));

        UpdateClothProperties();

        for (int i = 0; i < deltaTimeDivideCount; i++)
        {
            // Damp Velocity && Update Positions
            pbdComputeShader.SetBuffer(pbdUpdatePositionsKernel, "PositionsIn", positionBufferPool[0]);
            pbdComputeShader.SetBuffer(pbdUpdatePositionsKernel, "PositionsOut", positionBufferPool[1]);
            pbdComputeShader.Dispatch(
                pbdUpdatePositionsKernel,
                threadGroupCount.x, 
                threadGroupCount.y, 
                threadGroupCount.z
            );

            for (int iter = 0; iter < distanceIterCount; iter++)
            {
                pbdComputeShader.SetBuffer(pbdProjectDistanceConstraintsKernel, "PositionsIn", positionBufferPool[1]);
                pbdComputeShader.SetBuffer(pbdProjectDistanceConstraintsKernel, "PositionsOut", positionBufferPool[2]);
                pbdComputeShader.Dispatch(
                    pbdProjectDistanceConstraintsKernel,
                    threadGroupCount.x, 
                    threadGroupCount.y, 
                    threadGroupCount.z
                );

                DispatchFixConstraintsKernel();
            }

            for (int iter = 0; iter < bendingIterCount; iter++)
            {
                pbdComputeShader.SetBuffer(pbdProjectBendingConstraintsKernel, "PositionsIn", positionBufferPool[1]);
                pbdComputeShader.SetBuffer(pbdProjectBendingConstraintsKernel, "PositionsOut", positionBufferPool[2]);
                pbdComputeShader.Dispatch(
                    pbdProjectBendingConstraintsKernel,
                    threadGroupCount.x, 
                    threadGroupCount.y, 
                    threadGroupCount.z
                );
            
                DispatchFixConstraintsKernel();
            }

            for (int iter = 0; iter < collisionIterCount; iter++)
            {
                foreach (var sphere in Spheres)
                {
                    Vector3 pos = sphere.transform.position;
                    float rad = sphere.transform.localScale[0] * 0.5f;
                    pbdComputeShader.SetFloat("sphereRadius", 1.1f * rad);
                    pbdComputeShader.SetVector("sphereCenter", new Vector4(pos.x, pos.y, pos.z, 1));
                    pbdComputeShader.SetBuffer(pbdProjectSphereCollisionsConstraintsKernel, "PositionsIn", positionBufferPool[1]);
                    pbdComputeShader.SetBuffer(pbdProjectSphereCollisionsConstraintsKernel, "PositionsOut", positionBufferPool[2]);
                    pbdComputeShader.Dispatch(
                        pbdProjectSphereCollisionsConstraintsKernel,
                        threadGroupCount.x, 
                        threadGroupCount.y, 
                        threadGroupCount.z
                    );
                }
                
                DispatchFixConstraintsKernel();
            }
        }
        
        normalComputeShader.Dispatch(normalComputeKernel, threadGroupCount.x, threadGroupCount.y, threadGroupCount.z);
        tangentComputeShader.Dispatch(tangentComputeKernel, threadGroupCount.x, threadGroupCount.y, threadGroupCount.z);
        
    }

    private void DispatchFixConstraintsKernel()
    {
        pbdComputeShader.SetBuffer(pbdProjectFixConstraintsKernel, "PositionsIn", positionBufferPool[2]);
        pbdComputeShader.SetBuffer(pbdProjectFixConstraintsKernel, "PositionsOut", positionBufferPool[1]);
        pbdComputeShader.Dispatch(
            pbdProjectFixConstraintsKernel,
            threadGroupCount.x, 
            threadGroupCount.y, 
            threadGroupCount.z
        );
    }

    private void UpdateMaterial()
    {
        
    }
    
    private void UpdateModelMatrix()
    {
        Transform tf = this.transform;
        Matrix4x4 mat = tf.localToWorldMatrix;
        while (tf.parent)
        {
            tf = tf.parent;
            mat = tf.localToWorldMatrix * mat;
        }

        modelMatrices[0] = mat;
        modelMatrices[1] = mat.inverse;
    }

    private void UpdateClothProperties()
    {
    }

    private void OnDestroy()
    {
        positionBufferPool[0].Release();
        positionBufferPool[1].Release();
        positionBufferPool[2].Release();
        texCoordBuffer.Release();
        indexBuffer.Release();
        normalBuffer.Release();
        tangentBuffer.Release();
    }

    private void OnRenderObject()
    {
        if (!hasInit)
            return;

        material.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Triangles, indexCount, 1);
        material.SetPass(1);                                               
        Graphics.DrawProceduralNow(MeshTopology.Triangles, indexCount, 1);
    }
}
