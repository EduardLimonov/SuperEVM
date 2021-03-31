#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "datcreater.cuh"

cudaError_t runGenerate();


int main()
{
    runGenerate();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runGenerate()
{
    int maxTriang = 100, maxCircl = 200, *dev_nTriang, *dev_nCircl, resTriang = 0, resCircl = 0;
    int nBlocks = 100, nThreads = 100;
    int polySize = 10000;

    int* dev_polygon;
    int* polygon = new int[polySize * polySize];

    int *dev_a = 0;
    Point d = { polySize / nThreads, polySize / nBlocks };
    Pair<thrust::device_vector<Rect>>* conflicts;
    thrust::device_vector<Rect>* rects;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&conflicts, nBlocks * nThreads * sizeof(Pair<thrust::device_vector<Rect>>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_polygon, polySize * polySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nTriang, 1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_nCircl, 1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&rects, 1 * sizeof(thrust::device_vector<Rect>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    create_rects<<<nBlocks, nThreads>>>(d, maxTriang, maxCircl, conflicts, polySize, polygon, rects);

    for (int i = 0; i < 4; i++)
        // чудовищная синхронизация, но всё потому, что недоступна CUDA 9
        resolveConflicts<<<nBlocks, nThreads >>> (conflicts, i);

    create_objects<<<nBlocks, nThreads>>> (maxTriang, maxCircl, conflicts, polySize, dev_polygon, rects, dev_nTriang, dev_nCircl);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(polygon, dev_polygon, polySize * polySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(&resTriang, dev_nTriang, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(&resCircl, dev_nCircl, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    std::cout << "TRIANGLES: " << resTriang << "\nCIRCLES: " << resCircl << std::endl;

Error:
    cudaFree(conflicts);
    cudaFree(dev_polygon);
    cudaFree(dev_nTriang);
    cudaFree(dev_nCircl);
    cudaFree(rects);
    
    return cudaStatus;
}
