/*
 * Copyright (C) 2021 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Nazareno Bruschi, UniBo (<nazareno.bruschi@unibo.it>)
 *          Francesco Conti,  UniBo (<f.conti@unibo.it>)
 */


/*
 * Depends from the model. Let's keep it simple and focus on the first conv layer.
 */


#ifndef NETWORK_IDS_C
#define NETWORK_IDS_C
extern int NETWORK_IDS[NB_LAYER];
#endif

#ifndef NETWORK_TYPES_C
#define NETWORK_TYPES_C
extern int NETWORK_TYPES[NB_LAYER];
#endif

#ifndef NETWORK_DIMS_C
#define NETWORK_DIMS_C
extern int NETWORK_DIMS[NB_LAYER][NB_DIM_ARGS];
#endif

extern unsigned int L3_input[NB_LAYER];
extern unsigned int L3_output[NB_LAYER];
extern unsigned int L3_param[NB_LAYER];
extern PI_L2 network_layer_t network_layers[NB_LAYER];
extern PI_L1 network_layer_t l1_layer;
extern struct pi_device cluster;