// ======================================================================== //
// Copyright 2009-2014 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "common/scene_subdiv_mesh.h"
#include "subdivpatch1base.h"

namespace embree
{

  /*! Construction from vertices and IDs. */
  SubdivPatch1Base::SubdivPatch1Base (const CatmullClarkPatch& ipatch,
					  const unsigned int gID,
					  const unsigned int pID,
					  const SubdivMesh *const mesh) 
      : geom(gID),
        prim(pID),  
        flags(0)
    {
      assert(sizeof(SubdivPatch1Base) == 5 * 64);

      u_range = Vec2f(0.0f,1.0f);
      v_range = Vec2f(0.0f,1.0f);

      /* init discrete edge tessellation levels and grid resolution */

      assert( ipatch.ring[0].edge_level >= 0.0f );
      assert( ipatch.ring[1].edge_level >= 0.0f );
      assert( ipatch.ring[2].edge_level >= 0.0f );
      assert( ipatch.ring[3].edge_level >= 0.0f );

      level[0] = max(ceilf(ipatch.ring[0].edge_level),1.0f);
      level[1] = max(ceilf(ipatch.ring[1].edge_level),1.0f);
      level[2] = max(ceilf(ipatch.ring[2].edge_level),1.0f);
      level[3] = max(ceilf(ipatch.ring[3].edge_level),1.0f);

      grid_u_res = max(level[0],level[2])+1; // n segments -> n+1 points
      grid_v_res = max(level[1],level[3])+1;
      grid_mask  = 0;

#if defined(__MIC__)
      grid_size_simd_blocks        = ((grid_u_res*grid_v_res+15)&(-16)) / 16;
      grid_subtree_size_64b_blocks = 5; // single leaf with u,v,x,y,z

      if (grid_size_simd_blocks == 1)
	{
	  mic_m m_active = 0xffff;
	  for (unsigned int i=grid_u_res-1;i<16;i+=grid_u_res)
	    m_active ^= (unsigned int)1 << i;
	  m_active &= ((unsigned int)1 << (grid_u_res * (grid_v_res-1)))-1;
	  grid_mask = m_active;
	}

#else
      /* 8-wide SIMD is default on Xeon */
      grid_size_simd_blocks        = ((grid_u_res*grid_v_res+7)&(-8)) / 8;
      grid_subtree_size_64b_blocks = (sizeof(Quad2x2)+63) / 64; // single Quad2x2
#endif
      /* need stiching? */

      const unsigned int int_edge_points0 = (unsigned int)level[0] + 1;
      const unsigned int int_edge_points1 = (unsigned int)level[1] + 1;
      const unsigned int int_edge_points2 = (unsigned int)level[2] + 1;
      const unsigned int int_edge_points3 = (unsigned int)level[3] + 1;

      if (int_edge_points0 < (unsigned int)grid_u_res ||
	  int_edge_points2 < (unsigned int)grid_u_res ||
	  int_edge_points1 < (unsigned int)grid_v_res ||
	  int_edge_points3 < (unsigned int)grid_v_res)
	flags |= TRANSITION_PATCH;
      

      /* has displacements? */
      if (mesh->displFunc != NULL)
	flags |= HAS_DISPLACEMENT;


      /* tessellate into grid blocks for larger grid resolutions, generate bvh4 subtree over grid blocks*/

      //if (grid_size_simd_blocks > 1)
        {
#if defined(__MIC__)
	  const size_t leafBlocks = 5;
#else
          const size_t leafBlocks = (sizeof(Quad2x2)+63) / 64;
#endif
          grid_subtree_size_64b_blocks = getSubTreeSize64bBlocks( leafBlocks ); // u,v,x,y,z 
        }

      /* determine whether patch is regular or not */

      if (ipatch.isRegular() && mesh->displFunc == NULL ) 
	{
	  flags |= REGULAR_PATCH;
	  patch.init( ipatch );
	}
      else
	{
	  GregoryPatch gpatch; 
	  gpatch.init( ipatch ); 
	  gpatch.exportDenseConrolPoints( patch.v );
	}
#if 0
      DBG_PRINT( grid_u_res );
      DBG_PRINT( grid_v_res );
      DBG_PRINT( grid_size_16wide_blocks );
      DBG_PRINT( grid_mask );
      DBG_PRINT( grid_subtree_size_64b_blocks );
#endif
    }

}