// ======================================================================== //
// Copyright 2009-2015 Intel Corporation                                    //
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

#include "bvh4.h"
#include "bvh4_refit.h"

#include "../builders/primrefgen.h"
#include "../builders/bvh_builder_sah.h"

#include "../../algorithms/parallel_for_for.h"
#include "../../algorithms/parallel_for_for_prefix_sum.h"

#include "../../common/subdiv/bezier_curve.h"
#include "../geometry/subdivpatch1cached_intersector1.h"

#include "../geometry/grid_aos.h"
#include "../geometry/subdivpatch1.h"
#include "../geometry/subdivpatch1cached.h"
#include "../geometry/grid_soa.h"

#include "../geometry/oriented.h"

namespace embree
{
  namespace isa
  {
    typedef FastAllocator::ThreadLocal2 Allocator;

    template<typename Primitive>
    struct CreateBVH4SubdivLeaf
    {
      __forceinline CreateBVH4SubdivLeaf (BVH4* bvh, PrimRef* prims) 
        : bvh(bvh), prims(prims) {}
      
      __forceinline int operator() (const BVHBuilderBinnedSAH::BuildRecord& current, Allocator* alloc)
      {
        size_t items = Primitive::blocks(current.prims.size());
        size_t start = current.prims.begin();
        Primitive* accel = (Primitive*) alloc->alloc1.malloc(items*sizeof(Primitive));
        BVH4::NodeRef node = bvh->encodeLeaf((char*)accel,items);
        for (size_t i=0; i<items; i++) {
          accel[i].fill(prims,start,current.prims.end(),bvh->scene,false);
        }
        *current.parent = node;
	return 1;
      }

      BVH4* bvh;
      PrimRef* prims;
    };

    struct BVH4SubdivPatch1BuilderBinnedSAHClass : public Builder
    {
      ALIGNED_STRUCT;

      BVH4* bvh;
      Scene* scene;
      mvector<PrimRef> prims;
      
      BVH4SubdivPatch1BuilderBinnedSAHClass (BVH4* bvh, Scene* scene)
        : bvh(bvh), scene(scene) {}

      void build(size_t, size_t) 
      {
        /* initialize all half edge structures */
        const size_t numPrimitives = scene->getNumPrimitives<SubdivMesh,1>();
        if (numPrimitives > 0 || scene->isInterpolatable()) {
          Scene::Iterator<SubdivMesh> iter(scene,scene->isInterpolatable());
          parallel_for(size_t(0),iter.size(),[&](const range<size_t>& range) {
              for (size_t i=range.begin(); i<range.end(); i++)
                if (iter[i]) iter[i]->initializeHalfEdgeStructures();
            });
        }

        /* skip build for empty scene */
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }
        bvh->alloc.reset();

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH4SubdivPatch1BuilderBinnedSAH");

        prims.resize(numPrimitives);
        auto progress = [&] (size_t dn) { bvh->scene->progressMonitor(dn); };
        auto virtualprogress = BuildProgressMonitorFromClosure(progress);
        const PrimInfo pinfo = createPrimRefArray<SubdivMesh,1>(scene,prims,virtualprogress);
        BVH4::NodeRef root;
        BVHBuilderBinnedSAH::build_reduce<BVH4::NodeRef>
          (root,BVH4::CreateAlloc(bvh),size_t(0),BVH4::CreateNode(bvh),BVH4::NoRotate(),CreateBVH4SubdivLeaf<SubdivPatch1>(bvh,prims.data()),virtualprogress,
           prims.data(),pinfo,BVH4::N,BVH4::maxBuildDepthLeaf,1,1,1,1.0f,1.0f);
        bvh->set(root,pinfo.geomBounds,pinfo.size());

	/* clear temporary data for static geometry */
	if (scene->isStatic()) {
          prims.clear();
          bvh->shrink();
        }
        bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims.clear();
      }
    };
    
    struct BVH4SubdivGridEagerBuilderBinnedSAHClass : public Builder
    {
      ALIGNED_STRUCT;

      BVH4* bvh;
      Scene* scene;
      mvector<PrimRef> prims;
      ParallelForForPrefixSumState<PrimInfo> pstate;
      
      BVH4SubdivGridEagerBuilderBinnedSAHClass (BVH4* bvh, Scene* scene)
        : bvh(bvh), scene(scene) {}

      void build(size_t, size_t) 
      {
        /* initialize all half edge structures */
        const size_t numPrimitives = scene->getNumPrimitives<SubdivMesh,1>();
        if (numPrimitives > 0 || scene->isInterpolatable()) {
          Scene::Iterator<SubdivMesh> iter(scene,scene->isInterpolatable());
          parallel_for(size_t(0),iter.size(),[&](const range<size_t>& range) {
              for (size_t i=range.begin(); i<range.end(); i++)
                if (iter[i]) iter[i]->initializeHalfEdgeStructures();
            });
        }

        /* skip build for empty scene */
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }
        bvh->alloc.reset();

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH4SubdivGridEagerBuilderBinnedSAH");

        auto progress = [&] (size_t dn) { bvh->scene->progressMonitor(dn); };
        auto virtualprogress = BuildProgressMonitorFromClosure(progress);

        /* initialize allocator and parallel_for_for_prefix_sum */
        Scene::Iterator<SubdivMesh> iter(scene);
        pstate.init(iter,size_t(1024));

        PrimInfo pinfo1 = parallel_for_for_prefix_sum( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        { 
          size_t p = 0;
          size_t g = 0;
          for (size_t f=r.begin(); f!=r.end(); ++f) {          
            if (!mesh->valid(f)) continue;
            patch_eval_subdivision(mesh->getHalfEdge(f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              float level[4]; SubdivPatch1Base::computeEdgeLevels(edge_level,subdiv,level);
              Vec2i grid = SubdivPatch1Base::computeGridSize(level);
              size_t N = GridAOS::getNumEagerLeaves(grid.x-1,grid.y-1);
              g+=N;
              p++;
            });
          }
          return PrimInfo(p,g,empty,empty);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo(a.begin+b.begin,a.end+b.end,empty,empty); });
        size_t numSubPatches = pinfo1.begin;
        if (numSubPatches == 0) {
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }

        prims.resize(pinfo1.end);
        if (pinfo1.end == 0) {
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }

        PrimInfo pinfo3 = parallel_for_for_prefix_sum( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        {
          FastAllocator::ThreadLocal& alloc = *bvh->alloc.threadLocal();
          
          PrimInfo s(empty);
          for (size_t f=r.begin(); f!=r.end(); ++f) {
            if (!mesh->valid(f)) continue;
            
            patch_eval_subdivision(mesh->getHalfEdge(f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              SubdivPatch1Base patch(mesh->id,f,subPatch,mesh,uv,edge_level,subdiv,vfloat::size);
              size_t N = GridAOS::createEager(patch,scene,mesh,f,alloc,&prims[base.end+s.end]);
              assert(N == GridAOS::getNumEagerLeaves(patch.grid_u_res-1,patch.grid_v_res-1));
              for (size_t i=0; i<N; i++)
                s.add(prims[base.end+s.end].bounds());
              s.begin++;
            });
          }
          return s;
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a, b); });

        PrimInfo pinfo(pinfo3.end,pinfo3.geomBounds,pinfo3.centBounds);
        
        BVH4::NodeRef root;
        BVHBuilderBinnedSAH::build_reduce<BVH4::NodeRef>
          (root,BVH4::CreateAlloc(bvh),size_t(0),BVH4::CreateNode(bvh),BVH4::NoRotate(),
           [&] (const BVHBuilderBinnedSAH::BuildRecord& current, Allocator* alloc) -> int {
             if (current.pinfo.size() != 1) THROW_RUNTIME_ERROR("bvh4_builder_subdiv: internal error");
             *current.parent = (size_t) prims[current.prims.begin()].ID();
             return 0;
           },
           progress,
           prims.data(),pinfo,BVH4::N,BVH4::maxBuildDepthLeaf,1,1,1,1.0f,1.0f);
        bvh->set(root,pinfo.geomBounds,pinfo.size());
        
	/* clear temporary data for static geometry */
	if (scene->isStatic()) {
          prims.clear();
          bvh->shrink();
        }
        bvh->cleanup();
        bvh->postBuild(t0);
      }

      void clear() {
        prims.clear();
      }
    };

	template<typename node_t>
	struct GenerateCompressedBVHLeaves {
		typedef node_t node;
		typedef FastAllocator::ThreadLocal Allocator;

		int count(std::vector<CatmullClarkPatch3fa> &patches) {
			return patches.size();
		}
		void generate_leaves(SubdivMesh* mesh, int geom, int prim, std::vector<CatmullClarkPatch3fa> &patches,
			PrimInfo &s, mvector<PrimRef> &prims, const PrimInfo &base, std::vector<Vec2f> &uvs,std::vector<int> &qlevel, Allocator& alloc) {
			for (int i = 0; i < patches.size(); i++) {
				auto p = patches[i];
				BBox3fa bounds = oriented::createCompressedBVHLeaf<node_t>(mesh, geom, prim, i, p, &prims[base.size() + s.size()], uvs,qlevel[i], alloc);
				s.add(bounds);
			}
		}
	};

    template<typename Leafgen> struct BVH4OrientedBuilderBinnedSAHClass : public Builder, protected Leafgen
    {
      ALIGNED_STRUCT;

      BVH4* bvh;
      Scene* scene;
      mvector<PrimRef> prims;
      ParallelForForPrefixSumState<PrimInfo> pstate;
      enum { fixed_subdiv_level = LEVEL_N +1};
      enum { count_only, construct_too };
      
      BVH4OrientedBuilderBinnedSAHClass (BVH4* bvh, Scene* scene)
        : bvh(bvh), scene(scene) {}


	  bool subdivide_patch_flat_rec(SubdivMesh* mesh, unsigned geomID, unsigned primID, const CatmullClarkPatch3fa &patch, int pIdx, const std::vector<Vec2f> &patch_uvs, int N, int level, std::vector<CatmullClarkPatch3fa> &flat_patches, std::vector<Vec2f> &flat_uvs, std::vector<Vec3f> &patch_normals, std::vector<int> &qlevel){
		  if (level >= N) {
		      //we are at the lowest level!
			  if (level >= 1) 
				  patch_normals.push_back(Vec3f());
			  return true;
		  }

		  array_t<CatmullClarkPatch3fa, 4> new_patches;
		  patch.subdivide(new_patches);
		  std::vector<Vec2f> new_uvs(16);
		  oriented::createUvs(patch_uvs, pIdx, new_uvs);
		  bool flatChild[4];
		  int nextLevel = level + 1;
		  std::vector<Vec3f> new_patch_normals;
		  for (int i = 0; i < 4; i++){
			  if (subdivide_patch_flat_rec(mesh, geomID, primID, new_patches[i], i, new_uvs, N, nextLevel, flat_patches, flat_uvs, new_patch_normals, qlevel)){
				  flatChild[i] = true;
			  }
			  else{
				  flatChild[i] = false;
			  } 
		  }
		  bool allChildrenFlat = flatChild[0] & flatChild[1] & flatChild[2] & flatChild[3];
		  if (allChildrenFlat){
			  for (int i = 0; i < 4; i++){
				  flat_patches.push_back(new_patches[i]);
				  qlevel.push_back(N - level);
				  flat_uvs.push_back(new_uvs[i * 4]);
				  flat_uvs.push_back(new_uvs[i * 4 + 1]);
				  flat_uvs.push_back(new_uvs[i * 4 + 2]);
				  flat_uvs.push_back(new_uvs[i * 4 + 3]);
			  }
		  }
		  else{
			  for (int i = 0; i < 4; i++) {
				  if (flatChild[i]){
					  flat_patches.push_back(new_patches[i]);
					  qlevel.push_back(N - level);
					  flat_uvs.push_back(new_uvs[i * 4]);
					  flat_uvs.push_back(new_uvs[i * 4 + 1]);
					  flat_uvs.push_back(new_uvs[i * 4 + 2]);
					  flat_uvs.push_back(new_uvs[i * 4 + 3]);
				  }
			  }
		  }
		  return false;

	  }

	  template<int M> std::vector<CatmullClarkPatch3fa> subdivide_patch_flat(SubdivMesh* mesh, unsigned geomId, unsigned patchId, GeneralCatmullClarkPatch3fa &patch, int N, std::vector<Vec2f> &flat_uvs, std::vector<int> &qlevel) {
		  std::vector<CatmullClarkPatch3fa> flat_patches;
		  array_t<CatmullClarkPatch3fa, GeneralCatmullClarkPatch3fa::SIZE> tmp1;
		  size_t n;
		  patch.subdivide(tmp1, n);
		  assert(n == 4);
		  
		  std::vector<Vec2f> first_uvs;
		  oriented::createFirstSubdUvs(first_uvs);


		  GeneralCatmullClarkPatch3fa::fix_quad_ring_order(tmp1);

		  std::vector<Vec3f> normals;
		  for (int i = 0; i < 4; i++){
			  subdivide_patch_flat_rec(mesh, geomId,patchId, tmp1[i],i,first_uvs, N, 2,flat_patches, flat_uvs,normals,qlevel);
		  }
		  
		  if (flat_patches.size() * 4 != flat_uvs.size())
			  std::cerr << "subd patches:" << flat_patches.size() << " and uvs " << flat_uvs.size() << "\n";
		  return flat_patches;
	  }





      void build(size_t, size_t) 
      {
        /* skip build for empty scene */
	const size_t numPrimitives = scene->getNumPrimitives<SubdivMesh,1>();
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }
        bvh->alloc.reset();

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH4SubdivGridEagerBuilderBinnedSAH");

        auto progress = [&] (size_t dn) { bvh->scene->progressMonitor(dn); };
        auto virtualprogress = BuildProgressMonitorFromClosure(progress);

        /* initialize all half edge structures */
        Scene::Iterator<SubdivMesh> iter(scene);
		
		//OBJMaterial* material = (OBJMaterial*)&g_ispc_scene->materials[materialID];
        for (size_t i=0; i<iter.size(); i++) // FIXME: parallelize
          if (iter[i]) iter[i]->initializeHalfEdgeStructures();
        
        /* initialize allocator and parallel_for_for_prefix_sum */
        pstate.init(iter,size_t(1024));

        PrimInfo pinfo 
	    = parallel_for_for_prefix_sum(pstate, iter, PrimInfo(empty), 
		[&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        	{
        	  size_t s = 0;
        	  for (size_t f=r.begin(); f!=r.end(); ++f) 
        	  {
        	    if (!mesh->valid(f)) continue;
		    auto half_edge = mesh->getHalfEdge(f);
		    const auto &vertex_buffer = mesh->getVertexBuffer();
		    GeneralCatmullClarkPatch3fa patch;
		    patch.init(half_edge, vertex_buffer.getPtr(), vertex_buffer.getStride());
			std::vector<Vec2f> uvs;
			std::vector<int> qlevel;
			std::vector<CatmullClarkPatch3fa> patches = subdivide_patch_flat<count_only>(mesh,mesh->id,f, patch, fixed_subdiv_level, uvs,qlevel);
		    s += Leafgen::count(patches);
        	  }
        	  return PrimInfo(s,empty,empty);
        	}, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo(a.size()+b.size(),empty,empty); }
	);

		//std::cout << "prepared: " << pinfo.size() << std::endl;
		//std::cout << "node_t size: " << sizeof(typename Leafgen::node) << std::endl;

        prims.resize(pinfo.size());
        if (pinfo.size() == 0) {
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }

		size_t dummy = 0;

        pinfo 
	    = parallel_for_for_prefix_sum(pstate, iter, PrimInfo(empty), 
		[&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
		{
		  FastAllocator::ThreadLocal& alloc = *bvh->alloc.threadLocal();
		  
		  PrimInfo s(empty);
		  for (size_t f=r.begin(); f!=r.end(); ++f) {
		    if (!mesh->valid(f)) continue;
		    auto half_edge = mesh->getHalfEdge(f);
		    const auto &vertex_buffer = mesh->getVertexBuffer();
		    // do subdivision
		    GeneralCatmullClarkPatch3fa patch;
		    patch.init(half_edge, vertex_buffer.getPtr(), vertex_buffer.getStride());

			std::vector<Vec2f> uvs;
			std::vector<int> qlevel;
			std::vector<CatmullClarkPatch3fa> patches = subdivide_patch_flat<construct_too>(mesh, mesh->id,f, patch, fixed_subdiv_level, uvs,qlevel);
		    Leafgen::generate_leaves(mesh,mesh->id, f, patches, s, prims, base,uvs,qlevel, alloc);
			++dummy;
		  }
		  return s;
		}, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); }
	);

	//size_t count = prims.size();
	//float rotSize = count * sizeof(LinearSpace3fa) / 1024.f / 1024.f;
	//float projSize = count * sizeof(Matrix3f) / 1024.f /  1024.f;
	//float clipSize = count * sizeof(BBox3fa) / 1024.f /  1024.f;
	//float pointerSize = count * (sizeof(typename Leafgen::node*) + sizeof(Vec2f*)) / 1024.f /  1024.f;
	//float uvSize = count * sizeof(Vec2f) * 2 / 1024.f / 1024.f;
	//float intSize = count * sizeof(int) * 2 / 1024.f / 1024.f;
	//float leafSize = count * sizeof(typename oriented::CompressedBVHLeaf<typename Leafgen::node>) / 1024.f / 1024.f;
	//float nodeSize = count * oriented::g_items * sizeof(typename Leafgen::node) / 1024.f / 1024.f;

	//std::cout << "Leaves: " << count << "\n" <<
	//			 "LeafSize: " << leafSize << "\n" <<
	//			 "sum: " << (rotSize + projSize + clipSize + pointerSize + uvSize + intSize) << "\n" <<
	//			 "nodes: " << nodeSize << std::endl;


	//std::cout << "used: " << pinfo.size() << "(" << dummy << ")" << std::endl;
	//std::cout << "size: " << prims.size() * sizeof(prims[0]) / 1024.f / 1024.f << std::endl;
	//std::cout << "primsize: " << sizeof(prims[0]) << std::endl;

        BVH4::NodeRef root;
        BVHBuilderBinnedSAH::build_reduce<BVH4::NodeRef>(root,BVH4::CreateAlloc(bvh),size_t(0),BVH4::CreateNode(bvh),BVH4::NoRotate(),
           [&] (const BVHBuilderBinnedSAH::BuildRecord& current, Allocator* alloc) -> int 
	   {
             if (current.pinfo.size() != 1) THROW_RUNTIME_ERROR("bvh4_builder_subdiv: internal error");
             *current.parent = (size_t) prims[current.prims.begin()].ID();
             return 0;
           },
           progress,
           prims.data(),pinfo,BVH4::N,BVH4::maxBuildDepthLeaf,1,1,1,1.0f,1.0f
	);
        bvh->set(root,pinfo.geomBounds,pinfo.size());
        
	/* clear temporary data for static geometry */
	bool staticGeom = scene->isStatic();
	if (staticGeom) prims.clear();
        bvh->alloc.cleanup();
        bvh->postBuild(t0);

      }

      void clear() {
        prims.clear();
      }
    };



    // =======================================================================================================
    // =======================================================================================================
    // =======================================================================================================

    struct BVH4SubdivPatch1CachedBuilderBinnedSAHClass : public Builder, public BVH4Refitter::LeafBoundsInterface
    {
      ALIGNED_STRUCT;

      BVH4* bvh;
      BVH4Refitter* refitter;
      Scene* scene;
      mvector<PrimRef> prims; 
      mvector<BBox3fa> bounds; 
      ParallelForForPrefixSumState<PrimInfo> pstate;
      size_t numSubdivEnableDisableEvents;

      BVH4SubdivPatch1CachedBuilderBinnedSAHClass (BVH4* bvh, Scene* scene)
        : bvh(bvh), refitter(nullptr), scene(scene), numSubdivEnableDisableEvents(0) {}
      
      virtual const BBox3fa leafBounds (BVH4::NodeRef& ref) const 
      {
        if (ref == BVH4::emptyNode) return BBox3fa(empty);
        size_t num; SubdivPatch1Cached *sptr = (SubdivPatch1Cached*)ref.leaf(num);
        const size_t index = ((size_t)sptr - (size_t)this->bvh->data_mem) / sizeof(SubdivPatch1Cached);
        return prims[index].bounds(); 
      }

      void build(size_t, size_t) 
      {
        /* initialize all half edge structures */
        bool fastUpdateMode = true;
        size_t numPrimitives = scene->getNumPrimitives<SubdivMesh,1>();
        if (numPrimitives > 0 || scene->isInterpolatable()) 
        {
          Scene::Iterator<SubdivMesh> iter(scene,scene->isInterpolatable());
          fastUpdateMode = parallel_reduce(size_t(0),iter.size(),true,[&](const range<size_t>& range)
          {
            bool fastUpdate = true;
            for (size_t i=range.begin(); i<range.end(); i++)
            {
              if (!iter[i]) continue;
              fastUpdate &= !iter[i]->vertexIndices.isModified(); 
              fastUpdate &= !iter[i]->faceVertices.isModified();
              fastUpdate &= !iter[i]->holes.isModified();
              fastUpdate &= !iter[i]->edge_creases.isModified();
              fastUpdate &= !iter[i]->edge_crease_weights.isModified();
              fastUpdate &= !iter[i]->vertex_creases.isModified();
              fastUpdate &= !iter[i]->vertex_crease_weights.isModified(); 
              fastUpdate &= iter[i]->levels.isModified();
              iter[i]->initializeHalfEdgeStructures();
              iter[i]->patch_eval_trees.resize(iter[i]->size());
            }
            return fastUpdate;
          }, [](const bool a, const bool b) { return a && b; });
        }

        /* only enable fast mode of no subdiv mesh got enabled or disabled since last run */
        fastUpdateMode &= numSubdivEnableDisableEvents == scene->numSubdivEnableDisableEvents;
        numSubdivEnableDisableEvents = scene->numSubdivEnableDisableEvents;

        /* skip build for empty scene */
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bounds.resize(numPrimitives);
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }
        if (!fastUpdateMode) 
          bvh->alloc.reset();

        double t0 = bvh->preBuild(TOSTRING(isa) "::BVH4SubdivPatch1CachedBuilderBinnedSAH");

        auto progress = [&] (size_t dn) { bvh->scene->progressMonitor(dn); };
        auto virtualprogress = BuildProgressMonitorFromClosure(progress);

        /* initialize allocator and parallel_for_for_prefix_sum */
        Scene::Iterator<SubdivMesh> iter(scene);
        pstate.init(iter,size_t(1024));
        PrimInfo pinfo = parallel_for_for_prefix_sum( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        { 
          size_t s = 0;
          for (size_t f=r.begin(); f!=r.end(); ++f) 
          {          
            if (!mesh->valid(f)) continue;
            s += patch_eval_subdivision_count (mesh->getHalfEdge(f)); 
            if (unlikely(!fastUpdateMode)) {
              auto alloc = [&] (size_t bytes) { return bvh->alloc.threadLocal()->malloc(bytes); };
              mesh->patch_eval_trees[f] = Patch3fa::create(alloc, mesh->getHalfEdge(f), mesh->getVertexBuffer().getPtr(), mesh->getVertexBuffer().getStride());
            }
          }
          return PrimInfo(s,empty,empty);
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo(a.size()+b.size(),empty,empty); });
        numPrimitives = pinfo.size();
        
        if (numPrimitives == 0) {
          prims.resize(numPrimitives);
          bounds.resize(numPrimitives);
          bvh->set(BVH4::emptyNode,empty,0);
          return;
        }
        
        prims.resize(numPrimitives);
        bounds.resize(numPrimitives);
        
        /* Allocate memory for gregory and b-spline patches */
        if (this->bvh->size_data_mem < sizeof(SubdivPatch1Cached) * numPrimitives) 
        {
          if (this->bvh->data_mem) os_free( this->bvh->data_mem, this->bvh->size_data_mem );
          this->bvh->data_mem      = nullptr;
          this->bvh->size_data_mem = 0;
        }
        
        if (bvh->data_mem == nullptr)
        {
          this->bvh->size_data_mem = sizeof(SubdivPatch1Cached) * numPrimitives;
          if ( this->bvh->size_data_mem != 0) this->bvh->data_mem = os_malloc( this->bvh->size_data_mem );        
          else                                this->bvh->data_mem = nullptr;
        }
        assert(this->bvh->data_mem);
        SubdivPatch1Cached *const subdiv_patches = (SubdivPatch1Cached *)this->bvh->data_mem;
        
        //atomic_t numChanged = 0;
        //atomic_t numUnchanged = 0;
        pinfo = parallel_for_for_prefix_sum( pstate, iter, PrimInfo(empty), [&](SubdivMesh* mesh, const range<size_t>& r, size_t k, const PrimInfo& base) -> PrimInfo
        {
          PrimInfo s(empty);
          for (size_t f=r.begin(); f!=r.end(); ++f) 
          {
            if (!mesh->valid(f)) continue;
            
            patch_eval_subdivision(mesh->getHalfEdge(f),[&](const Vec2f uv[4], const int subdiv[4], const float edge_level[4], int subPatch)
            {
              const unsigned int patchIndex = base.size()+s.size();
              assert(patchIndex < numPrimitives);
              SubdivPatch1Base& patch = subdiv_patches[patchIndex];
              BBox3fa bound = empty;
              
              if (likely(fastUpdateMode)) {
                bool grid_changed = patch.updateEdgeLevels(edge_level,subdiv,mesh,vfloat::size);
                //grid_changed = true;
                //if (grid_changed) atomic_add(&numChanged,1); else atomic_add(&numUnchanged,1);
                if (grid_changed) {
                  patch.resetRootRef();
                  bound = evalGridBounds(patch,0,patch.grid_u_res-1,0,patch.grid_v_res-1,patch.grid_u_res,patch.grid_v_res,mesh);
                }
                else {
                  patch.updateRootRef(scene->commitCounter+1);
                  bound = bounds[patchIndex];
                }
              }
              else {
                new (&patch) SubdivPatch1Cached(mesh->id,f,subPatch,mesh,uv,edge_level,subdiv,vfloat::size);
                bound = evalGridBounds(patch,0,patch.grid_u_res-1,0,patch.grid_v_res-1,patch.grid_u_res,patch.grid_v_res,mesh);
                //patch.root_ref.data = (int64_t) GridSOA::create(&patch,scene,[&](size_t bytes) { return (*bvh->alloc.threadLocal())(bytes); });
              }
              bounds[patchIndex] = bound;
              prims[patchIndex] = PrimRef(bound,patchIndex);
              s.add(bound);
            });
          }
          return s;
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a, b); });
        //PRINT(numChanged);
        //PRINT(numUnchanged);

        if (fastUpdateMode)
        {
          if (refitter == nullptr)
            refitter = new BVH4Refitter(bvh,*(BVH4Refitter::LeafBoundsInterface*)this);

          refitter->refit();
        }
        else
        {
          BVH4::NodeRef root;
          BVHBuilderBinnedSAH::build_reduce<BVH4::NodeRef>
            (root,BVH4::CreateAlloc(bvh),size_t(0),BVH4::CreateNode(bvh),BVH4::NoRotate(),
             [&] (const BVHBuilderBinnedSAH::BuildRecord& current, Allocator* alloc) -> int {
              size_t items = current.pinfo.size();
              assert(items == 1);
              const unsigned int patchIndex = prims[current.prims.begin()].ID();
              SubdivPatch1Cached *const subdiv_patches = (SubdivPatch1Cached *)this->bvh->data_mem;
              *current.parent = bvh->encodeLeaf((char*)&subdiv_patches[patchIndex],1);
              return 0;
            },
             progress,
             prims.data(),pinfo,BVH4::N,BVH4::maxBuildDepthLeaf,1,1,1,1.0f,1.0f);

          bvh->set(root,pinfo.geomBounds,pinfo.size());
          delete refitter; refitter = nullptr;
        }
        
	/* clear temporary data for static geometry */
	if (scene->isStatic()) {
          prims.clear();
          bvh->shrink();
        }
        bvh->cleanup();
        bvh->postBuild(t0);
      }
      
      void clear() {
        prims.clear();
      }
    };
    
    /* entry functions for the scene builder */
    Builder* BVH4SubdivPatch1BuilderBinnedSAH   (void* bvh, Scene* scene, size_t mode) { return new BVH4SubdivPatch1BuilderBinnedSAHClass((BVH4*)bvh,scene); }
    Builder* BVH4SubdivGridEagerBuilderBinnedSAH   (void* bvh, Scene* scene, size_t mode) { return new BVH4SubdivGridEagerBuilderBinnedSAHClass((BVH4*)bvh,scene); }
    Builder* BVH4SubdivPatch1CachedBuilderBinnedSAH   (void* bvh, Scene* scene, size_t mode) { return new BVH4SubdivPatch1CachedBuilderBinnedSAHClass((BVH4*)bvh,scene); }

	// oriented
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_FullPrecision (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::ref>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_QuantizedUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::uni332n>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_QuantizedNonUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::man332n>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_CompressedUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::uni332a>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_CompressedNonUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::man332a>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_HalfSlabUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::uni332b>>((BVH4*)bvh,scene); }
	Builder* BVH4SubdivOrientedBuilderBinnedSAH_HalfSlabNonUniform (void* bvh, Scene* scene, size_t node) { return new BVH4OrientedBuilderBinnedSAHClass<GenerateCompressedBVHLeaves<oriented::man332b>>((BVH4*)bvh,scene); }

  }
}
