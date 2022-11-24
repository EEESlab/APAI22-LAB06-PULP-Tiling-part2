#include "main.h"
#include "network_desc.h"
/*
 * Convolution kernel
 */
void convolution_run()
{
  /*
   * PULP-NN Convolution kernel
   */
   pulp_nn_conv(
    l1_layer.input_data,
    l1_layer.buffer_0,
    NULL,
    l1_layer.output_data,
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
void kernel_init(int tileH, int tileW, int tileC)
{
#if defined(DEBUG)
  printf("---> Entering Kernel Initialization...\n");
#endif

  pi_cl_dma_cmd_t dma_cmd;

#ifndef EXERCISE2
  pi_cl_dma_cmd_2d(
    (unsigned int)(network_layers[0].input_data + (l1_layer.layer_dim.c_in * tileC) + (network_layers[0].layer_dim.c_in * l1_layer.layer_dim.x_in * tileW) + (network_layers[0].layer_dim.c_in * network_layers[0].layer_dim.x_in * l1_layer.layer_dim.y_in * tileH)),
    (unsigned int)l1_layer.input_data, 
    l1_layer.layer_dim.c_in * l1_layer.layer_dim.x_in * l1_layer.layer_dim.y_in,
    network_layers[0].layer_dim.c_in * network_layers[0].layer_dim.x_in,
    l1_layer.layer_dim.c_in * l1_layer.layer_dim.x_in,
    PI_CL_DMA_DIR_EXT2LOC,
    &dma_cmd);
  pi_cl_dma_cmd_wait(&dma_cmd);
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
void kernel_run()
{
  /*
   * Fork the job over available cores
   */
  pi_cl_team_fork(NB_CORES, convolution_run, NULL);
}

/*
 * Move back the output results in L2
 */
void kernel_end(int tileH, int tileW, int tileC)
{
#if defined(DEBUG)
  printf("---> Entering Kernel Ending...\n");
#endif

  pi_cl_dma_cmd_t dma_cmd;
#ifndef EXERCISE2
  pi_cl_dma_cmd_2d(
    (unsigned int)(network_layers[0].output_data + (l1_layer.layer_dim.c_out * tileC) + (network_layers[0].layer_dim.c_out * l1_layer.layer_dim.x_out * tileW) + (network_layers[0].layer_dim.c_out * network_layers[0].layer_dim.x_out * l1_layer.layer_dim.y_out * tileH)),
    (unsigned int)l1_layer.output_data, 
    l1_layer.layer_dim.c_out * l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out,
    network_layers[0].layer_dim.c_out * network_layers[0].layer_dim.x_out,
    l1_layer.layer_dim.c_out * l1_layer.layer_dim.x_out,
    PI_CL_DMA_DIR_LOC2EXT,
    &dma_cmd);

  pi_cl_dma_cmd_wait(&dma_cmd);
#endif

#if defined(DEBUG)
  printf("---> Exiting Kernel Ending...\n");
#endif
}