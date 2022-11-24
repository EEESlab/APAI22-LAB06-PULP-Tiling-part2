#include "main.h"
#include "network_desc.h"
/*
 * Buffer in which open the files
 */
 
int network_init()
{

#if defined(DEBUG)
  printf("-> Entering Network Initialization...\n");
#endif

  for(int i=0; i<NB_LAYER; i++)
  {
    network_layers[i].id        = LAYER_ID(i);
    network_layers[i].type      = LAYER_TYPE(i);
    network_layers[i].layer_dim = LAYER_DIM(i);
  }

#if defined(DEBUG)
  printf("-> Exiting Network Initialization...\n");
#endif

  return 0;
}

int network_free()
{
  pi_l2_free(network_layers[0].output_data, (unsigned int)(network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.y_out * network_layers[0].layer_dim.c_out * sizeof(unsigned char)));
  return 0;
}