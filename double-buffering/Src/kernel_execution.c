#include "main.h"
#include "network_desc.h"
#include "dims.h"
#include "dory_dma.h"
/*
 * Convolution kernel
 */
void convolution_run(int buffer_idx)
{
  /*
   * PULP-NN Convolution kernel
   */
   pulp_nn_conv(
    l1_layer.input_data + buffer_idx * l1_layer.layer_dim.c_in * l1_layer.layer_dim.x_in * l1_layer.layer_dim.y_in,
    l1_layer.buffer_0,
    NULL,
    l1_layer.output_data + buffer_idx * l1_layer.layer_dim.c_out * l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out,
    l1_layer.param_data,
    8,
    l1_layer.layer_dim.x_in,
    l1_layer.layer_dim.y_in,
    l1_layer.layer_dim.c_in,
    l1_layer.layer_dim.x_out,
    l1_layer.layer_dim.y_out,    
    l1_layer.layer_dim.c_out,
    l1_layer.layer_dim.x_ker,
    l1_layer.layer_dim.y_ker,
    l1_layer.layer_dim.top_pad,
    l1_layer.layer_dim.bot_pad,
    l1_layer.layer_dim.lef_pad,
    l1_layer.layer_dim.rig_pad,
    l1_layer.layer_dim.x_stride,
    l1_layer.layer_dim.y_stride);
}

/*
 * Copy data in L1
 */
void kernel_init(int  tileH_idx, int  tileW_idx, int  tileC_idx, int buffer_idx)
{
#if defined(DEBUG)
  printf("---> Entering Kernel Initialization...\n");
#endif

#ifdef USE_L1_MEM
  if (tileH_idx == 0 && tileW_idx == 0 && tileC_idx == 0)
    dory_dma_init();

  /** Task 5.2. Tile overlapping
   *
   *  Calculate the tile overlapping and adjust the `network_offset` variable to account for that.
   */
  const int network_offset =  tileH_idx * l1_layer.layer_dim.y_in * network_layers[0].layer_dim.x_in * network_layers[0].layer_dim.c_in
      +  tileW_idx * l1_layer.layer_dim.x_in * network_layers[0].layer_dim.c_in
      +  tileC_idx * l1_layer.layer_dim.c_in;
  const int ext_stride_1d = network_layers[0].layer_dim.c_in * network_layers[0].layer_dim.x_in;
  const int buffer_size = l1_layer.layer_dim.c_in * l1_layer.layer_dim.x_in * l1_layer.layer_dim.y_in;
  const int double_buffer_offset = buffer_idx * buffer_size;

  dory_dma_copy copy = {
    .ext = (void *)(network_layers[0].input_data + network_offset),
    .loc = (void *)l1_layer.input_data + double_buffer_offset, 
    .stride_2d = 1,  // unused in 2D transfers
    .number_of_2d_copies = 1,  // only 1 2D copy in 2D transfers
    .stride_1d = ext_stride_1d,
    .number_of_1d_copies = l1_layer.layer_dim.y_in,
    .length_1d_copy = l1_layer.layer_dim.x_in * l1_layer.layer_dim.c_in,
    .dir = DORY_DMA_DIR_EXT2LOC
  };

  dory_dma_memcpy_2d_async(&copy);
#else
  l1_layer.input_data = network_layers[0].input_data;
#endif

#if defined(DEBUG)
  printf("---> Exiting Kernel Initialization...\n");
#endif
}

/*
 * Execute the convolution kernel
 */
void kernel_run(int buffer_idx)
{
  /*
   * Fork the job over available cores
   */
  pi_cl_team_fork(NB_CORES, convolution_run, buffer_idx);
}

/*
 * Move back the output results in L2
 */
void kernel_end(int  tileH_idx, int  tileW_idx, int  tileC_idx, int buffer_idx)
{
#if defined(DEBUG)
  printf("---> Entering Kernel Ending...\n");
#endif

#ifdef USE_L1_MEM
  dory_dma_copy copy = {
    .ext = (void *)(network_layers[0].output_data + (l1_layer.layer_dim.c_out *  tileC_idx) + (network_layers[0].layer_dim.c_out * l1_layer.layer_dim.x_out *  tileW_idx) + (network_layers[0].layer_dim.c_out * network_layers[0].layer_dim.x_out * l1_layer.layer_dim.y_out *  tileH_idx)),
    .loc = (void *)l1_layer.output_data + buffer_idx * l1_layer.layer_dim.c_out * l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out, 
    .stride_2d = 1,  // unused in 2D transfers
    .number_of_2d_copies = 1,  // only 1 2D copy in 2D transfers
    .number_of_1d_copies = l1_layer.layer_dim.y_out,
    .length_1d_copy = l1_layer.layer_dim.x_out * l1_layer.layer_dim.c_out,
    .stride_1d = network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.c_out,
    .dir = DORY_DMA_DIR_LOC2EXT
  };

  dory_dma_memcpy_2d_async(&copy);
#endif

#if defined(DEBUG)
  printf("---> Exiting Kernel Ending...\n");
#endif
}

void kernel_wait()
{
  dory_dma_wait();
#if defined(DEBUG)
  printf("---> Waiting Kernel Transfer...\n");
#endif
}
