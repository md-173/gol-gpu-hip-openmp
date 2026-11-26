#define INCLUDE_CPU_VERSION
#include "game_of_life.c"
#include <hip/hip_runtime.h>
#define BLOCK_SIZE 16

// Exit with file/line on HIP errors 
// Use HIP_CHECK_ERROR on hipMalloc, hipMemcpy, and kernel-launch
void __hip_check_error(hipError_t err, const char *file, int line){
    if(err != hipSuccess){
        fprintf(stderr, "HIP error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}

#define HIP_CHECK_ERROR(X)({\
    __hip_check_error((X), __FILE__, __LINE__);\
})

// GPU version of cpu_game_of_life_step
// HIP kernel, one thread per cell, instead of CPU versions nested for loops
__global__ void gpu_game_of_life_step(int *current_grid, int *next_grid, int n, int m){
    int neighbours = 0;
    int n_i[8], n_j[8];

    // Threads calculate i, j from their block and thread ID
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // If more threads than cells the extras skip
    if (i < n && j < m) {
        
        // Same as CPU version for rest of function
        n_i[0] = i - 1; n_j[0] = j - 1;
        n_i[1] = i - 1; n_j[1] = j;
        n_i[2] = i - 1; n_j[2] = j + 1;
        n_i[3] = i;     n_j[3] = j + 1;
        n_i[4] = i + 1; n_j[4] = j + 1;
        n_i[5] = i + 1; n_j[5] = j;
        n_i[6] = i + 1; n_j[6] = j - 1;
        n_i[7] = i;     n_j[7] = j - 1;

        if(n_i[0] >= 0 && n_j[0] >= 0 && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
        if(n_i[1] >= 0 && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
        if(n_i[2] >= 0 && n_j[2] < m && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
        if(n_j[3] < m && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
        if(n_i[4] < n && n_j[4] < m && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
        if(n_i[5] < n && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
        if(n_i[6] < n && n_j[6] >= 0 && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
        if(n_j[7] >= 0 && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

        if(current_grid[i*m + j] == ALIVE && (neighbours == 2 || neighbours == 3)){
            next_grid[i*m + j] = ALIVE;
        } else if(current_grid[i*m + j] == DEAD && neighbours == 3){
            next_grid[i*m + j] = ALIVE;
        } else {
            next_grid[i*m + j] = DEAD;
        }
    }
}

int* gpu_game_of_life(const int *initial_state, int n, int m, int nsteps, float *kernel_time_ms){   
 
    // Two device read/write buffers  
    int *dev_grid, *dev_updated_grid;
    HIP_CHECK_ERROR(hipMalloc(&dev_grid, sizeof(int) * n * m));
    HIP_CHECK_ERROR(hipMalloc(&dev_updated_grid, sizeof(int) * n * m));

    // Device kernel timing with hipEvents
    hipEvent_t kernel_start, kernel_stop;
    HIP_CHECK_ERROR(hipEventCreate(&kernel_start));
    HIP_CHECK_ERROR(hipEventCreate(&kernel_stop));

    // Host buffer for the final grid
    int *grid = (int *) malloc(sizeof(int) * n * m);
    if(!grid){
        printf("Error while allocating memory.\n");
        exit(1);
    }
    
    int current_step = 0;
    int *tmp = NULL;
    
    // Send the initial state to the GPU
    HIP_CHECK_ERROR(hipMemcpy(dev_grid, initial_state, sizeof(int) * n * m, hipMemcpyHostToDevice));

    // Grid of BLOCK_SIZE x BLOCK_SIZE blocks, one thread per cell    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Start kernel timer
    HIP_CHECK_ERROR(hipEventRecord(kernel_start));
 
    while(current_step != nsteps){
        current_step++;
        
        // To visualise each step uncomment the two lines below, debug only
        // HIP_CHECK_ERROR(hipMemcpy(grid, dev_grid, sizeof(int) * n * m, hipMemcpyDeviceToHost));
        // visualise(VISUAL_ASCII, current_step, grid, n, m);
    
        // One parallel GPU step, check async errors    
        gpu_game_of_life_step<<<dimGrid, dimBlock>>>(dev_grid, dev_updated_grid, n, m);
        HIP_CHECK_ERROR(hipGetLastError());
 
        // Swap pointers, no GPU copy
        tmp = dev_grid;
        dev_grid = dev_updated_grid;
        dev_updated_grid = tmp;

    }
    // Stop kernel timer
    HIP_CHECK_ERROR(hipEventRecord(kernel_stop));

    // Copy result to host
    HIP_CHECK_ERROR(hipMemcpy(grid, dev_grid, sizeof(int) * n * m, hipMemcpyDeviceToHost));

    // Total kernel time of all steps
    HIP_CHECK_ERROR(hipEventElapsedTime(kernel_time_ms, kernel_start, kernel_stop));

    // Free device buffers
    HIP_CHECK_ERROR(hipFree(dev_grid));
    HIP_CHECK_ERROR(hipFree(dev_updated_grid));
    
    // Destroy events
    HIP_CHECK_ERROR(hipEventDestroy(kernel_start));
    HIP_CHECK_ERROR(hipEventDestroy(kernel_stop));
    return grid;
}

int main(int argc, char **argv)
{
    struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
    getinput(argc, argv, opt);
    int n = opt->n, m = opt->m, nsteps = opt->nsteps;
    int *initial_state = (int *) malloc(sizeof(int) * n * m);
    if(!initial_state){
        printf("Error while allocating memory.\n");
        return -1;
    }
    generate_IC(opt->iictype, initial_state, n, m);

    // Run GPU version
    float kernel_time_ms = 0.0f;
    struct timeval gpu_start = init_time();
    int *gpu_final_state = gpu_game_of_life(initial_state, n, m, nsteps, &kernel_time_ms);
    float Tp = get_elapsed_time(gpu_start);    

    // Run CPU version for correctness check & timing
    struct timeval cpu_start = init_time();
    int *cpu_final_state = cpu_game_of_life(initial_state, n, m, nsteps);
    float Ts = get_elapsed_time(cpu_start);

    // Correctness check, GPU & CPU must have the same final state grids
    if(memcmp(gpu_final_state, cpu_final_state, sizeof(int) * n * m) == 0) {
        printf("Correct Output, GPU matches CPU\n");
    } else {
        printf("Incorrect Output, GPU != CPU\n");
    }

    // Performance Results
    printf("CPU Time (Ts) = %.3f ms\n", Ts);
    printf("GPU Time (Tp) = %.3f ms\n", Tp);
    printf("Speedup S (Ts/Tp) = %.3f\n", Ts / Tp);
    printf("t_k (%% of time in kernel) = %.2f %%\n", (kernel_time_ms / Tp) * 100.0f);

    free(gpu_final_state);
    free(cpu_final_state);
    free(initial_state);
    free(opt);
    return 0;
}
