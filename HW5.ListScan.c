// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
cudaError_t err = stmt;                                               \
if (err != cudaSuccess) {                                             \
wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
return -1;                                                        \
}                                                                     \
} while(0)

__global__ void blockScan(float* input, float* output, int len)
{
    __shared__ float data[2*BLOCK_SIZE];
    int tx  = threadIdx.x;
    int tx2 = tx+BLOCK_SIZE;
    int  x  = 2*blockDim.x*blockIdx.x + tx;
    int  x2 = x+BLOCK_SIZE;
    
    data[tx]  = x  < len ? input[x]  : 0.;
    data[tx2] = x2 < len ? input[x2] : 0.;
    
    for (int stride=1; stride<=BLOCK_SIZE; stride*=2)
    {
        __syncthreads();
        int idx = (tx+1)*2*stride-1;
        if (idx<2*BLOCK_SIZE)
            data[idx] += data[idx-stride];
    }
    
    for (int stride=BLOCK_SIZE/2; stride>0; stride/=2)
    {
        __syncthreads();
        int idx = (tx+1)*2*stride-1;
        if (idx+stride<2*BLOCK_SIZE)
            data[idx+stride] += data[idx];
    }
    
    __syncthreads();
    if (x<len)
        output[x]  = data[tx];
    if (x2<len)
        output[x2] = data[tx2];
}
//
__global__ void getO1(float* output, float* O1, int len)
{
    if (blockIdx.x==0)
    {
        int tx = threadIdx.x;
        int x  = tx*2*BLOCK_SIZE-1;
        O1[tx] = (x>0 && x<len) ? output[x] : 0.;
    }
}
//
__global__ void addVal(float* vec, float* valToAdd, int len)
{
    int val = valToAdd[blockIdx.x];
    int x   = 2*blockDim.x*blockIdx.x + threadIdx.x;
    int x2  = x + BLOCK_SIZE;
    
    if (x<len)
        vec[x] +=val;
    if (x2<len)
        vec[x2]+=val;
}
//
int main(int argc, char ** argv)
{
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * deviceO1;
    float * deviceO2;
    
    int numElements; // number of elements in the list
    
    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput  = (float*) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The number of input elements in the input is ", numElements);
    int nbBlocks = numElements / (2*BLOCK_SIZE);
    nbBlocks    += numElements % (2*BLOCK_SIZE)==0 ? 0 : 1;
    
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput , numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceO1    , nbBlocks*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceO2    , nbBlocks*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(deviceO1, 0, nbBlocks*sizeof(float)));
    wbCheck(cudaMemset(deviceO2, 0, nbBlocks*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");
    
    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 blocks (nbBlocks, 1, 1);
    dim3 threads(BLOCK_SIZE , 1, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    blockScan<<<blocks, threads>>>(deviceInput, deviceOutput, numElements);
    getO1    <<<blocks, threads>>>(deviceOutput, deviceO1, numElements);
    blockScan<<<blocks, threads>>>(deviceO1, deviceO2, nbBlocks);
    addVal   <<<blocks, threads>>>(deviceOutput, deviceO2, numElements);
    
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceO1);
    cudaFree(deviceO2);
    wbTime_stop(GPU, "Freeing GPU Memory");
    
    wbSolution(args, hostOutput, numElements);
    
    free(hostInput);
    free(hostOutput);
    
    return 0;
}



