/*
 * Copyright (C) 2021 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alessio Burrello, UniBo (<alessio.burrello@unibo.it>)
 *          Nazareno Bruschi, UniBo (<nazareno.bruschi@unibo.it>)
 *          Davide Nadalini,  UniBo (<d.nadalini@unibo.it>)
 *          Francesco Conti,  UniBo (<f.conti@unibo.it>)
 *          Luka Macan,       UniBo (<luka.macan@unibo.it>)
 */

#include "main.h"
#include "dims.h"

int NETWORK_IDS[NB_LAYER] = {0};
int NETWORK_TYPES[NB_LAYER] = {CONV};

int NETWORK_DIMS[NB_LAYER][NB_DIM_ARGS] = {{INPUT_WIDTH, //x_in
                                            INPUT_HEIGHT, //y_in
                                            INPUT_CHANNEL, //c_in
                                            OUTPUT_WIDTH, //x_out
                                            OUTPUT_HEIGHT, //y_out
                                            OUTPUT_CHANNEL, //c_out
                                            1,  //x_stride
                                            1,  //y_stride
                                            0,  //top_pad
                                            0,  //bot_pad
                                            0,  //lef_pad
                                            0,  //rig_pad
                                            WEIGHTS_KERNEL_WIDTH,  //x_ker
                                            WEIGHTS_KERNEL_HEIGHT}};//y_ker
PI_L2 network_layer_t network_layers[NB_LAYER];
PI_L1 network_layer_t l1_layer;
struct pi_device cluster;

/*
 * Cluster initialization
 */
void cluster_init()
{
#if defined(DEBUG)
  printf("-> Entering Cluster Initialization...\n");
#endif
  /*
   * Alloc L1 space for tiles: BE CAREFUL -> L1 is very limited (64kB)
   */
  l1_layer.id   = LAYER_ID(0);
  l1_layer.type = LAYER_TYPE(0);
  
  /*
   * Local dimensions determines the tile strategy
   */
  l1_layer.layer_dim.c_in     = network_layers[0].layer_dim.c_in;
  l1_layer.layer_dim.c_out    = network_layers[0].layer_dim.c_out;
  l1_layer.layer_dim.x_ker    = network_layers[0].layer_dim.x_ker;
  l1_layer.layer_dim.y_ker    = network_layers[0].layer_dim.y_ker;
  l1_layer.layer_dim.top_pad  = network_layers[0].layer_dim.top_pad;
  l1_layer.layer_dim.bot_pad  = network_layers[0].layer_dim.bot_pad;
  l1_layer.layer_dim.lef_pad  = network_layers[0].layer_dim.lef_pad;
  l1_layer.layer_dim.rig_pad  = network_layers[0].layer_dim.rig_pad;
  l1_layer.layer_dim.x_stride = network_layers[0].layer_dim.x_stride;
  l1_layer.layer_dim.y_stride = network_layers[0].layer_dim.y_stride;

#ifdef ENABLE_TILING
  const int TILING_PARAMETER =  2/* COMPLETE EX1 */;
#else
  const int TILING_PARAMETER =  1;
#endif

  l1_layer.layer_dim.x_out = network_layers[0].layer_dim.x_out / TILING_PARAMETER;
  l1_layer.layer_dim.y_out = network_layers[0].layer_dim.y_out / TILING_PARAMETER;
  l1_layer.layer_dim.x_in  = l1_layer.layer_dim.x_out + WEIGHTS_KERNEL_WIDTH - 1;
  l1_layer.layer_dim.y_in  = l1_layer.layer_dim.y_out + WEIGHTS_KERNEL_HEIGHT - 1;

  /*
   * PUT YOUR CODE HERE
   */

#ifdef USE_L1_MEM
  l1_layer.input_data  = pi_l1_malloc(&cluster, (unsigned int)(l1_layer.layer_dim.x_in  * l1_layer.layer_dim.y_in  * l1_layer.layer_dim.c_in * sizeof(unsigned char) * 2));
  l1_layer.param_data  = pi_l1_malloc(&cluster, (unsigned int)(l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * l1_layer.layer_dim.c_out * sizeof(signed char)));
  l1_layer.output_data = pi_l1_malloc(&cluster, (unsigned int)(l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out * l1_layer.layer_dim.c_out * sizeof(unsigned char) * 2));
#endif

  /*
   * PULP-NN Conv kernel exploits im2col to reorder the input data: Take it into account for L1 space
   */
  if(l1_layer.type == CONV)
  {
#ifdef USE_L1_MEM
  	l1_layer.buffer_0 = pi_l1_malloc(&cluster, (unsigned int)(2 * l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * NB_CORES * sizeof(unsigned char)));
#else
  	l1_layer.buffer_0 = pi_l2_malloc((unsigned int)(2 * l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * NB_CORES * sizeof(unsigned char)));
#endif
  }

#if defined(DEBUG)
  printf("-> Exiting Cluster Initialization...\n");
#endif
}

/*
 * FC application entry point
 */
int main()
{
  printf("\nEntering Main. Checking for Exercise...\n");
#if defined EXERCISE4
    printf("Executing Exercise 4\n");
#elif defined EXERCISE5
    printf("Executing Exercise 5\n");    
#else
  printf("No Exercise selected. Exiting...\n");
  return;
#endif

  /*
   * Initialize the network and copy data in L3
   */
  int err = network_init();
  if(err)
  {
    return err;
  }

  /*
   * Copy data of the single layer in L2 (if it is possible)
   */
  err = layer_init();
  if(err)
  {
    return err;
  }

  /*
   * Now you should have all the parameters in L2 to compute the output of the first layer
   */
  /* 
   * DESCRIPTION:
   * PULP Cluster has 8 identical cores that can share and compute heavy computational workload in parallel.
   * PULP Cluster is a PMSIS device. Open it and send the task to initialize it.
   */

  struct pi_cluster_conf cluster_conf;
  pi_cluster_conf_init(&cluster_conf);
  pi_open_from_conf(&cluster, &cluster_conf);
  if(pi_cluster_open(&cluster))
  {
    return -CLUSTER_ERROR;
  }

  struct pi_cluster_task cluster_task;
  pi_cluster_send_task_to_cl(&cluster, pi_cluster_task(&cluster_task, cluster_init, NULL));
  
  pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_INSTR);
  pi_perf_reset();
  pi_perf_start();
  
  pi_cluster_send_task_to_cl(&cluster, pi_cluster_task(&cluster_task, layer_run, NULL));
  
  pi_perf_stop();
  uint32_t instr_cnt  = pi_perf_read(PI_PERF_INSTR);
  uint32_t cycles_cnt = pi_perf_read(PI_PERF_CYCLES);
#if defined(PERFORMANCE)
  printf("[FC]: instructions = %d, tot_cycles = %d\n", instr_cnt, cycles_cnt);
#endif
  err = layer_check();
  if(err)
  {
  	return err;
  }

  err = layer_free();
  if(err)
  {
    return err;
  }

  err = network_free();
  if(err)
  {
    return err;
  }

#if defined(DEBUG)
  printf("Exiting Main...\n");
#endif
  return 0;
}
