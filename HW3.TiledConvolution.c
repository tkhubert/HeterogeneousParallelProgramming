#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
cudaError_t err = stmt;                                               \
if (err != cudaSuccess) {                                             \
wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
return -1;                                                        \
}                                                                     \
} while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE
namespace
{
    const int NB_CHANNELS  = 3;
    const int O_TILE_WIDTH = 12;
    const int I_TILE_WIDTH = O_TILE_WIDTH + Mask_width -1;
}
//
// called with
// dim3 blocks((imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1);
// dim3 threads(I_TILE_WIDTH, I_TILE_WIDTH, 1);
__global__ void convolution(float* iImageData, float* oImageData, const float* __restrict__ maskData,
                            int imageWidth, int imageHeight, int imageChannels, int maskRows, int maskColumns)
{
    __shared__ float imgData[I_TILE_WIDTH][I_TILE_WIDTH][NB_CHANNELS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int oRow = blockIdx.y*O_TILE_WIDTH+ty;
    int oCol = blockIdx.x*O_TILE_WIDTH+tx;
    int iRow = oRow - Mask_radius;
    int iCol = oCol - Mask_radius;
    
    for (int c=0; c<imageChannels; ++c)
    {
        bool check = (iRow>=0 && iRow<imageHeight && iCol>=0 && iCol<imageWidth);
        imgData[ty][tx][c] = check ? iImageData[(iRow*imageWidth+iCol)*imageChannels+c] : 0.;
    }
    __syncthreads();
    
    if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH && oRow<imageHeight && oCol<imageWidth)
    {
        for (int c=0; c<imageChannels; ++c)
        {
            float output=0.;
            for (int i=0; i<maskRows; ++i)
                for (int j=0; j<maskColumns; ++j)
                    output += imgData[ty+i][tx+j][c]*maskData[i*maskColumns + j];
            output = min(max(output, 0.), 1.);
            
            oImageData[(oRow*imageWidth+oCol)*imageChannels+c] = output;
        }
    }
}
//


int main(int argc, char* argv[])
{
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    
    args = wbArg_read(argc, argv); /* parse the input arguments */
    
    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile  = wbArg_getInputFile(args, 1);
    
    inputImage   = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);
    
    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */
    
    imageWidth    = wbImage_getWidth(inputImage);
    imageHeight   = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    
    hostInputImageData  = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData , imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    
    
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    
    
    wbTime_start(Compute, "Doing the computation on the GPU");
    
    //@@ INSERT CODE HERE
    dim3 blocks((imageWidth-1)/O_TILE_WIDTH+1, (imageHeight-1)/O_TILE_WIDTH+1, 1);
    dim3 threads(I_TILE_WIDTH, I_TILE_WIDTH, 1);
    convolution<<<blocks, threads>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight, imageChannels, maskRows, maskColumns);
    
    wbTime_stop(Compute, "Doing the computation on the GPU");
    
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    wbSolution(args, outputImage);
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
    
    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    
    return 0;
}


