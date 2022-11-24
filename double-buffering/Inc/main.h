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
 */


#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "bsp/bsp.h"

// #define EXERCISE1
//#define EXERCISE2
// #define EXERCISE3
#define EXERCISE4

#define USE_L1_MEM
#define ENABLE_TILING

#define GENERAL_ERROR    1
#define FLASH_ERROR      2
#define FILESYSTEM_ERROR 3
#define RAM_ERROR        4
#define FILE_ERROR       5
#define L2_ERROR         6
#define CLUSTER_ERROR    7
#define CHECK_ERROR      8

void cluster_init();
int layer_check();
int layer_init();
int layer_free();
void layer_run();
int layer_check();
int network_init();
int network_free();

#ifndef NB_LAYER
#define NB_LAYER 1
#endif

#ifndef NB_DIM_ARGS
#define NB_DIM_ARGS 14
#endif


#ifndef PERF_MEASUREMENT
#define PERF_MEASUREMENT
// #define DEBUG
#define PERFORMANCE
#endif

/*
 * Usefuf structures to handle the network objects (i.e. layers)
 */
/* Type of layer */
#ifndef layer_type_e
#define layer_type_e
typedef enum layer_type_e
{
 CONV,
 DEPTH,
 LINEAR,
 POOLING,
 RELU,
 SOFTMAX
}layer_type_t;
#endif

/* Dimensions of the layer */
#ifndef layer_type_s
#define layer_type_s
typedef struct layer_dim_s
{
  int x_in;
  int y_in;
  int c_in;
  int x_out;
  int y_out;
  int c_out;
  int x_stride;
  int y_stride;
  int top_pad;
  int bot_pad;
  int lef_pad;
  int rig_pad;
  int x_ker;
  int y_ker;
} layer_dim_t;
#endif

/* network information and data pointers */
#ifndef network_layer_s
#define network_layer_s
typedef struct network_layer_s
{
  int id;
  layer_type_t type;
  layer_dim_t layer_dim;
  unsigned char *input_data;
  unsigned char *output_data;
  signed char *param_data;
  unsigned char *buffer_0;
  unsigned char *buffer_1;
} network_layer_t;
#endif

#ifndef GLOBAL_VARIABLES
#define GLOBAL_VARIABLES
#define LAYER_ID(x)    NETWORK_IDS[(x)]
#define LAYER_TYPE(x)  NETWORK_TYPES[(x)]
#define LAYER_DIM(x)   *(layer_dim_t *)NETWORK_DIMS[(x)]
#endif
