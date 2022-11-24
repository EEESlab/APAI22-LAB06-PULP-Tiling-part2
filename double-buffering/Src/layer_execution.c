#include "main.h"
#include "network_desc.h"
#include "input.h"
#include "weights.h"
#include "output.h"

int layer_init()
{
#if defined(DEBUG)
  printf("-> Entering Layer Initialization...\n");
#endif

  network_layers[0].input_data = input;
  if(network_layers[0].input_data == NULL)
  {
    return -L2_ERROR;
  }

  network_layers[0].param_data = weights;
  if(network_layers[0].param_data == NULL)
  {
    return -L2_ERROR;
  }

  network_layers[0].output_data = pi_l2_malloc((unsigned int)(network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.y_out * network_layers[0].layer_dim.c_out * sizeof(unsigned char)));
  if(network_layers[0].output_data == NULL)
  {
    return -L2_ERROR;
  }

#if defined(DEBUG)
  printf("-> Exiting Layer Initialization...\n");
#endif

  return 0;
}

/*
 * Cluster application entry point
 */
void layer_run()
{
#if defined(DEBUG)
  printf("--> Entering Layer Running...\n");
#endif

  pi_cl_dma_cmd_t dma_cmd;

  int dma_copy_size = l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * l1_layer.layer_dim.c_out;

  /*
   * Parameters copy: Let's keep it simple, avoid the tiling along channels. Move all the parameters in L1
   */
#ifdef USE_L1_MEM
  pi_cl_dma_cmd((unsigned int)network_layers[0].param_data, (unsigned int)l1_layer.param_data, dma_copy_size, PI_CL_DMA_DIR_EXT2LOC, &dma_cmd);
  pi_cl_dma_cmd_wait(&dma_cmd);
#else
  l1_layer.param_data = network_layers[0].param_data;
#endif

#ifndef USE_L1_MEM
  l1_layer.output_data = network_layers[0].output_data;
#endif
  /*
   * Tile loop bounds
   */
  int nb_h_tile = 0;
  int nb_w_tile = 0;

  /*
   * Tile loop indexes
   */
  int h_tile = 0;
  int w_tile = 0;

  /*
   * Cluster performance counters
   */
#if defined(PERFORMANCE)
  pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES | 1 << PI_PERF_CYCLES | 1 << PI_PERF_INSTR);
  pi_perf_reset();
  pi_perf_start();
#endif

#ifdef ENABLE_TILING
  /** EXERCISE5: compute the number of tiles on the output size
   */
  /**
   * these are the variables that store the layers' sizes
   * Network's layers:
   *    network_layers[0].layer_dim.x_in  --> input w dimension
   *    network_layers[0].layer_dim.y_in  --> input h dimension
   *    network_layers[0].layer_dim.x_out --> output w dimension
   *    network_layers[0].layer_dim.y_out --> output h dimension
   * L1 layer:
   *    l1_layer.layer_dim.x_in  --> input w dimension
   *    l1_layer.layer_dim.y_in  --> input h dimension
   *    l1_layer.layer_dim.x_out --> output w dimension
   *    l1_layer.layer_dim.y_out --> output h dimension
   */

  /** Task 5.3. 3x3 convolution number of tiles
   *
   *  Check the calculation of number of tiles for the 3x3 convolution.
   *  HINT: Look at the way the layer size is calculated in main.c lines 73-76
   */

  nb_h_tile = network_layers[0].layer_dim.y_in / l1_layer.layer_dim.y_in;
  nb_w_tile = network_layers[0].layer_dim.x_in / l1_layer.layer_dim.x_in;
#else
  nb_h_tile = 1;
  nb_w_tile = 1;
#endif 


  /** EXERCISE4.1: implement double buffering
   *  the sequence to obtain double buffering would be:
   *    - initialize buffer#0
   *    - wait until initialization buffer#0 is done
   *    - initialize buffer#1
   *    - run inference on buffer#0
   * 
   * You can use these functions:
   *  - void kernel_init(int tileH_idx, int tileW_idx, int tileC, int buffer_idx)   -->  starts DMA transfer (L2->L1) for a specific tile (you can select which one with the indexes)
   *  - void kernel_run(int buffer_idx)                                             -->  Runs inference for the selected buffer_idx
   *  - void kernel_end(int tileH_idx, int tileW_idx, int tileC, int buffer_idx)    -->  DMA copies the the output from L1 to L2 memory
   *  - void kernel_wait()                                                          -->  waits until next DMA interrupt (the interrupt tells you the dma trasnfer has been completed)
   */
  int buffer_idx = 0;
  int next_w_tile = 0;
  int next_h_tile = 0;
  /*
  * Kernel initialization for tile 0 (outside the for loop)
  */
  /* YOUR CODE HERE */;

  for(h_tile=0; h_tile < nb_h_tile; h_tile++) 
  {
    for(w_tile=0; w_tile < nb_w_tile; w_tile++) 
    {

      buffer_idx += 1;
      buffer_idx = buffer_idx % 2;

      /*
       * Kernel wait, until previus DMA transfer has been completed
       */
      /* YOUR CODE HERE */;

      /*
      * Kernel initialization for the next tile
      */
      next_w_tile = (w_tile + 1) % nb_w_tile;
      next_h_tile = h_tile + (next_w_tile == 0 ? 1 : 0);  // If the next_w_tile is 0, that means we went into a new row, so the next_h_tile is h_tile + 1
      if (( nb_h_tile * h_tile + w_tile ) != ( nb_h_tile * nb_w_tile - 1 ))
        /* YOUR CODE HERE */;

      /*
       * Executing the main kernel
       */
       
      /* YOUR CODE HERE */;


      /*
       * Kernel ending
       */
      /* YOUR CODE HERE */;


    }
  }
  /*
  * Kernel wait, until previus DMA transfer has been completed
  */  
  /* YOUR CODE HERE */;
#if defined(PERFORMANCE)
  pi_perf_stop();
  uint32_t instr_cnt      = pi_perf_read(PI_PERF_INSTR);
  uint32_t cycles_cnt     = pi_perf_read(PI_PERF_CYCLES);
  uint32_t act_cycles_cnt = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
  printf("[0]: instructions = %d, tot_cycles = %d, active_cycles = %d \n", instr_cnt, cycles_cnt, act_cycles_cnt);
#endif

#if defined(DEBUG)
  printf("--> Exiting Layer Running...\n");
#endif
}

/*
 * Check if the outputs are the same of the golden model
 */
int layer_check()
{
  int tot_layer_out_dim = network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.y_out * network_layers[0].layer_dim.c_out;

  int errors = 0;
  for(int i=0; i<tot_layer_out_dim; i++)
  {
    if(output[i] != network_layers[0].output_data[i])
    {
      printf("ERROR at index %d, expected %x and got %x\n", i, output[i], network_layers[0].output_data[i]);
      errors++;
    }
  }

  if (errors != 0) {
    printf("Received %d errors\n", errors);
    return -GENERAL_ERROR;
  }

  printf("Exiting layer with 0 errors\n");

  return 0;
}

int layer_free()
{
#ifdef USE_L1_MEM
  pi_l1_free(&cluster, l1_layer.input_data , (unsigned int)(l1_layer.layer_dim.x_in  * l1_layer.layer_dim.y_in  * l1_layer.layer_dim.c_in * sizeof(unsigned char)));
  pi_l1_free(&cluster, l1_layer.param_data , (unsigned int)(l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * l1_layer.layer_dim.c_out * sizeof(signed char)));
  pi_l1_free(&cluster, l1_layer.output_data, (unsigned int)(l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out * l1_layer.layer_dim.c_out * sizeof(unsigned char)));
#endif
  pi_cluster_close(&cluster);
  return 0;
}
