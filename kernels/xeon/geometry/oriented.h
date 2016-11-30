#pragma once

#include "../../../common/math/linearspace3.h"
#include "../../common/subdiv/feature_adaptive_eval.h"
#include "../bvh4/bvh4_intersector_node.h"

#include "discrete_tessellation.h"

#include "oriented_node.h"


#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

using namespace Eigen;


// enables rojection of patches
#define PROJECTION

// force orthogonal space (else sheered space is used)
//#define ORTHOGONAL

// exit when fist hitpoint in cbvh found
#define EARLY_OUT

// use vertex grid for on-the-fly triangle build
//#define WITH_GRID

#define LEVEL_N 3
#define LEVEL_Q 4

namespace embree {
	namespace oriented {

		// uncompressed nodes, only quantization
		typedef Node<flavor::non,quantization::uni,quantization::uni,3,3,2> uni332n;
		typedef Node<flavor::non,quantization::man,quantization::man2,3,3,2> man332n;

		// compressed nodes + quantization
		typedef Node<flavor::com,quantization::uni,quantization::uni,3,3,2> uni332a;
		typedef Node<flavor::com,quantization::man,quantization::man2,3,3,2> man332a;

		typedef Node<flavor::mid,quantization::zero,quantization::uni,3,3,2> uni332b;
		typedef Node<flavor::mid,quantization::zero,quantization::man2,3,3,2> man332b;

		// uncompressed unqantizded node
		typedef Node<flavor::ref,quantization::zero,quantization::zero,32,32,32> ref;

		const static size_t g_edges = static_cast<size_t>(pow(2.f, LEVEL_Q));
		const static float g_rcpEdges = 1.f / static_cast<float>(g_edges);
		const static size_t g_items = static_cast<float>((pow(4.f, LEVEL_Q)-1)/3);
		const static float g_epsilon = 1.0E-4F;

		//decoder code :
		//https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
		// "Insert" a 0 bit after each of the 16 low bits of x
		//
		inline uint32_t Part1By1(uint32_t x) {
			x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
			x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
			x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
			x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
			x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
			return x;
		}

		inline uint32_t EncodeMorton2(uint32_t x, uint32_t y) {
			return (Part1By1(y) << 1) + Part1By1(x);
		}

		inline uint32_t Compact1By1(uint32_t x) {
			x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
			x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
			x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
			x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
			x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
			return x;

		}

		inline uint32_t DecodeMorton2X(uint32_t code)
		{
			  return Compact1By1(code >> 0);
		}

		inline uint32_t DecodeMorton2Y(uint32_t code)
		{
			  return Compact1By1(code >> 1);
		}

		//  XY projection
		inline Matrix3f ComputeLinearEstimate(const std::vector<Vector2f>& input, const std::vector<Vector2f>& target) {
			Matrix3f M;
			M.setZero();


			MatrixXd A(8,8);

			for (size_t i = 0; i < 4; ++i) {

				const Vector2f& p = target[i];
				const Vector2f& q = input[i];

				A.row(0+i) << q[0], q[1], 1.f, 0.f, 0.f, 0.f, -q[0]*p[0], -q[1]*p[0];
				A.row(4+i) << 0.f, 0.f, 0.f, q[0], q[1], 1.f, -q[0]*p[1], -q[1]*p[1];
			}

			VectorXd b(8);
			for (size_t i = 0; i < 2; ++i)
				for (size_t k = 0; k < 4; ++k)
					b.row(i*4+k) << target[k][i];

			VectorXd x(8);

			x = A.fullPivLu().solve(b);

			M <<  x[0],  x[1], x[2],
			  	  x[3],  x[4], x[5],
			  	  x[6],  x[7],  1.f;

			return M;
		}

		inline Vec3f project(const Vec3f& a, const Matrix3f& proj) {
			Vector3f tmp = Vector3f(a.x, a.y, 1.f);
			tmp = proj * tmp;
			return Vec3f(tmp.x() / tmp.z(), tmp.y() / tmp.z(), a.z);
		}

		///////////////////

		inline void createFirstSubdUvs(std::vector<Vec2f> &uvs){
			Vec2f uv0 = Vec2f(0, 0);
			Vec2f uv1 = Vec2f(1, 0);
			Vec2f uv2 = Vec2f(1, 1);
			Vec2f uv3 = Vec2f(0, 1);
			Vec2f mid = 0.25f*(uv3 + uv0 + uv1 + uv2);
			uvs.resize(16);
			int s = 1;
			uvs[0] = uv0;			uvs[1] = 0.5f*(uv1 + uv0);			uvs[2] = mid;			uvs[3] = 0.5f*(uv0 + uv3);
			uvs[s * 4 + 0] = 0.5f*(uv1 + uv0);	uvs[s * 4 + 1] = uv1;	uvs[s * 4 + 2] = (uv1 + uv2)*0.5f;	uvs[s * 4 + 3] = mid;
			s++;
			uvs[s * 4 + 0] = mid;	uvs[s * 4 + 1] = 0.5f*(uv1 + uv2);	uvs[s * 4 + 2] = uv2;	uvs[s * 4 + 3] = 0.5f*(uv2 + uv3);
			s++;
			uvs[s * 4 + 0] = 0.5f*(uv0 + uv3);	uvs[s * 4 + 1] = mid;	uvs[s * 4 + 2] = 0.5f*(uv3 + uv2);	uvs[s * 4 + 3] = uv3;



		}

		inline void createUvs(const std::vector<Vec2f> &in_uvs, int pIdx, std::vector<Vec2f> &new_uvs){
			const Vec2f &uv0 = in_uvs[pIdx * 4 + 0];
			const Vec2f &uv1 = in_uvs[pIdx * 4 + 1];
			const Vec2f &uv2 = in_uvs[pIdx * 4 + 2];
			const Vec2f &uv3 = in_uvs[pIdx * 4 + 3];

			Vec2f mid = 0.25f*(uv0 + uv1 + uv2 + uv3);

			int s = 1;
			new_uvs[0] = uv0;			new_uvs[1] = 0.5f*(uv1 + uv0);			new_uvs[2] = mid;			new_uvs[3] = 0.5f*(uv0 + uv3);
			new_uvs[s * 4 + 0] = 0.5f*(uv1 + uv0);	new_uvs[s * 4 + 1] = uv1;	new_uvs[s * 4 + 2] = (uv1 + uv2)*0.5f;	new_uvs[s * 4 + 3] = mid;
			s++;
			new_uvs[s * 4 + 0] = mid;	new_uvs[s * 4 + 1] = 0.5f*(uv1 + uv2);	new_uvs[s * 4 + 2] = uv2;	new_uvs[s * 4 + 3] = 0.5f*(uv2 + uv3);
			s++;
			new_uvs[s * 4 + 0] = 0.5f*(uv0 + uv3);	new_uvs[s * 4 + 1] = mid;	new_uvs[s * 4 + 2] = 0.5f*(uv3 + uv2);	new_uvs[s * 4 + 3] = uv3;


		}

		template<typename Patch>
		static BBox3fa displaceVertices(SubdivMesh* mesh, unsigned geomID, unsigned primID, Patch &patch, const float* us, const float* vs, Vec3f &v00, Vec3f &v01, Vec3f &v10, Vec3f &v11){
			
			Vec3f n00 = normalize(cross(v10 - v00, v01 - v00));
			Vec3f n01 = normalize(cross(v00 - v01, v11 - v01));
			Vec3f n10 = normalize(cross(v11 - v10, v00 - v10));
			Vec3f n11 = normalize(cross(v01 - v11, v10 - v11));

			float verticesX[4];
			float verticesY[4];
			float verticesZ[4];
			verticesX[0] = v00.x; verticesY[0] = v00.y; verticesZ[0] = v00.z;
			verticesX[1] = v01.x; verticesY[1] = v01.y; verticesZ[1] = v01.z;
			verticesX[2] = v10.x; verticesY[2] = v10.y; verticesZ[2] = v10.z;
			verticesX[3] = v11.x; verticesY[3] = v11.y; verticesZ[3] = v11.z;
			
			float normalsX[4];
			float normalsY[4];
			float normalsZ[4];
			normalsX[0] = n00.x; normalsY[0] = n00.y; normalsZ[0] = n00.z;
			normalsX[1] = n01.x; normalsY[1] = n01.y; normalsZ[1] = n01.z;
			normalsX[2] = n10.x; normalsY[2] = n10.y; normalsZ[2] = n10.z;
			normalsX[3] = n11.x; normalsY[3] = n11.y; normalsZ[3] = n11.z;

			if (mesh->displFunc){
				mesh->displFunc(mesh->userPtr, geomID, primID, us, vs, &normalsX[0], &normalsY[0], &normalsZ[0],
					&verticesX[0], &verticesY[0], &verticesZ[0], 4);
			}
			v00.x = verticesX[0]; v00.y = verticesY[0]; v00.z = verticesZ[0];
			v01.x = verticesX[1]; v01.y = verticesY[1]; v01.z = verticesZ[1];
			v10.x = verticesX[2]; v10.y = verticesY[2]; v10.z = verticesZ[2];
			v11.x = verticesX[3]; v11.y = verticesY[3]; v11.z = verticesZ[3];

			BBox3fa bounds(v00);
			bounds.extend(v01);
			bounds.extend(v10);
			bounds.extend(v11);

			return bounds;
		}

		inline bool intersect_triangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2, Ray &ray) {
			const float &a_x = v0.x;
			const float &a_y = v0.y;
			const float &a_z = v0.z;
			const float &a = a_x - v1.x;
			const float &b = a_y - v1.y;
			const float &c = a_z - v1.z;
			const float &d = a_x - v2.x;
			const float &e = a_y - v2.y;
			const float &f = a_z - v2.z;
			const float &g = ray.dir.x;
			const float &h = ray.dir.y;
			const float &i = ray.dir.z;
			const float &j = a_x - ray.org.x;
			const float &k = a_y - ray.org.y;
			const float &l = a_z - ray.org.z;

			float common1 = e*i - h*f;
			float common2 = g*f - d*i;
			float common3 = d*h - e*g;
			float M 	= a * common1  +  b * common2  +  c * common3;
			float beta 	= j * common1  +  k * common2  +  l * common3;

			common1 = a*k - j*b;
			common2 = j*c - a*l;
			common3 = b*l - k*c;
			float gamma = i * common1  +  h * common2  +  g * common3;
			float tt	= -(f * common1  +  e * common2  +  d * common3);

			beta /= M;
			gamma /= M;
			tt /= M;

			if (tt >= 0.f && tt < ray.tfar && tt >= ray.tnear)
				if (beta >= 0.f && gamma >= 0.f && beta + gamma <= 1.f)
				{
					ray.tfar = tt;
					ray.v = beta;
					ray.u = gamma;
					return true;
				}

			return false;
		}

		// Intersects triangles build on the fly from grid vertices starrting with idx
		
		inline bool intersect_triangles(const int idx, const Vec3f* grid, Ray& ray)  {
			const float u = static_cast<float>(DecodeMorton2X(idx)) * g_rcpEdges;
			const float v = static_cast<float>(DecodeMorton2Y(idx)) * g_rcpEdges;

			// Decode indizes for Triangle vertices
			const size_t mIdx = static_cast<size_t>(static_cast<size_t>((v * g_edges) * (g_edges+1)) + u * g_edges);
			const size_t mIdx2 = mIdx+1;
			const size_t mIdx3 = mIdx2+g_edges;
			const size_t mIdx4 = mIdx3+1;

			const Vec3f& v0 = grid[mIdx];
			const Vec3f& v1 = grid[mIdx2];
			const Vec3f& v2 = grid[mIdx3];
			const Vec3f& v3 = grid[mIdx4];

			const bool hit1 = intersect_triangle(v1, v3, v0, ray);
			const bool hit2 = intersect_triangle(v2, v0, v3, ray);


			if (hit2) {
				ray.u = u + ray.u * g_rcpEdges;
				ray.v = v + (1.f - ray.v) * g_rcpEdges;
				return true;
			}

			if (hit1) {
				ray.u = u + (1.f - ray.u) * g_rcpEdges;
				ray.v = v + ray.v * g_rcpEdges;
				return true;
			}

			return false;
		}



		inline bool intersect_clip(const BBox3f &box, const Vec3fa& ray_origin, const Vec3fa& ray_dir, float &t_near, float &t_far) {

			const Vec3fa ray_rdir = rcp_safe(ray_dir);
			const Vec3fa ray_org_rdir = ray_origin*ray_rdir;

			const Vec3fa lower = msub(box.lower, ray_rdir, ray_org_rdir);
			const Vec3fa upper = msub(box.upper, ray_rdir, ray_org_rdir);

			t_near = fmaxf(t_near, reduce_max(min(lower, upper)));
			t_far = fminf(t_far, reduce_min(max(lower, upper)));

			return t_near <= t_far;
		}


		//This is our Transition node
		template<typename node_t>
			struct CompressedBVHLeaf {
				int geom_id, prim_id;
				Vec2f uvs[2];
				node_t* nodes;

				LinearSpace3f space;
				BBox3f box;
#ifdef PROJECTION
				Matrix3f proj;
#endif

#ifdef WITH_GRID
				Vec3f * grid;
#endif

			};

		template<typename Patch>
		BBox3fa getDisplacedPositions(SubdivMesh* mesh, unsigned geomID, unsigned primID, const std::vector<Patch> &patches, const std::vector<Vec2f> &uvs, std::vector<Vec3f> &displ_positions){
			BBox3fa overallAABounds = empty;
			for (int i = 0; i < patches.size(); i++){
				//iterate over all patches and evaluate displaced positions of the vertices.
				Vec3f v00 = patches[i].ring[0].vtx;
				Vec3f v10 = patches[i].ring[1].vtx;
				Vec3f v11 = patches[i].ring[2].vtx;
				Vec3f v01 = patches[i].ring[3].vtx;

				float u[4]; u[0] = uvs[i * 4].x; u[1] = uvs[i * 4 + 3].x; u[2] = uvs[i * 4 + 1].x; u[3] = uvs[i * 4 + 2].x;
				float v[4]; v[0] = uvs[i * 4].y; v[1] = uvs[i * 4 + 3].y; v[2] = uvs[i * 4 + 1].y; v[3] = uvs[i * 4 + 2].y;

				//comput displaced bounds and extend the overall AABB of the current leaf.
				BBox3fa bounds = displaceVertices(mesh, geomID, primID, patches[i], u, v, v00, v01, v10, v11);
				displ_positions.push_back(v00);
				displ_positions.push_back(v10);
				displ_positions.push_back(v01);
				displ_positions.push_back(v11);
				overallAABounds.extend(v00);
				overallAABounds.extend(v01);
				overallAABounds.extend(v10);
				overallAABounds.extend(v11);
			}
			return overallAABounds;
		}

		template<typename Patch>
		std::vector<Patch>* maxSubdivide(Patch &patch, size_t maxLevel) {
			std::vector<std::vector<Patch>*> patches;
			patches.resize(2);
			patches.clear();
			patches.push_back(new std::vector<Patch>());
			patches.push_back(new std::vector<Patch>());
			patches[0]->push_back(patch);

			array_t<Patch, 4> tmpPatches;

			size_t src;
			size_t tar;
			for (size_t i = 0; i < maxLevel; ++i) {
				src = i % 2;
				tar = 1 - src;
				patches[tar]->clear();

				for (auto it = patches[src]->begin(); it != patches[src]->end(); ++it) {
					it->subdivide(tmpPatches);
					patches[tar]->push_back(tmpPatches[0]);
					patches[tar]->push_back(tmpPatches[1]);
					patches[tar]->push_back(tmpPatches[3]); // changed order
					patches[tar]->push_back(tmpPatches[2]); // changed order
				}
			}
			delete patches[src];
			return patches[tar];
		};

		template<typename node_t>
		void buildHierarchy(CompressedBVHLeaf<node_t>& leaf, std::vector<Vec3f>& vertices) {
			std::vector<node_t> nodes;

			std::vector<std::vector<BBox3fa>> boxHierarchy;
			boxHierarchy.emplace_back(std::vector<BBox3fa>());

			const LinearSpace3f& space = leaf.space;
#ifdef PROJECTION
			const Matrix3f& proj = leaf.proj;
#endif


			for (size_t i = 0; i < vertices.size(); i+=4) {
#ifdef PROJECTION
				boxHierarchy[0].emplace_back(BBox3fa(project(xfmPoint(space, vertices[i+0]), proj)));
				boxHierarchy[0].back().extend(project(xfmPoint(space, vertices[i+1]), proj));
				boxHierarchy[0].back().extend(project(xfmPoint(space, vertices[i+2]), proj));
				boxHierarchy[0].back().extend(project(xfmPoint(space, vertices[i+3]), proj));
#else
				boxHierarchy[0].emplace_back(BBox3fa(xfmPoint(space, vertices[i+0])));
				boxHierarchy[0].back().extend(xfmPoint(space, vertices[i+1]));
				boxHierarchy[0].back().extend(xfmPoint(space, vertices[i+2]));
				boxHierarchy[0].back().extend(xfmPoint(space, vertices[i+3]));
#endif
			} 

			while(boxHierarchy.back().size() > 1) {
				std::vector<BBox3fa>& curr = boxHierarchy.back();
				std::vector<BBox3fa> currLevel;
				for (size_t i = 0; i < curr.size(); i+=4)
					currLevel.push_back(merge(curr[i+0], curr[i+1], curr[i+2], curr[i+3]));
				boxHierarchy.push_back(currLevel);
			}

			std::reverse(boxHierarchy.begin(), boxHierarchy.end());
#ifndef PROJECTION
			leaf.box = BBox3f(boxHierarchy[0][0].lower, boxHierarchy[0][0].upper);
#endif

			for (size_t i = 0; i < boxHierarchy.size() - 1; ++i)
				for (size_t k = 0; k < boxHierarchy[i].size(); ++k) {
					nodes.emplace_back(node_t());
					nodes.back().setAABB(boxHierarchy[i][k],
										  boxHierarchy[i+1][k*4+0],
										  boxHierarchy[i+1][k*4+1],
										  boxHierarchy[i+1][k*4+2],
										  boxHierarchy[i+1][k*4+3]);

						for (size_t l = 0; l < 4; ++l) {
							BBox3fa quantBox;
							nodes.back().getAABB(quantBox, boxHierarchy[i][k], l);
							if ( i < boxHierarchy.size() - 2)
								boxHierarchy[i+1][k*4+l] = quantBox;
						}
				}

			for (size_t i = 0; i < nodes.size(); ++i)
				leaf.nodes[i] = nodes[i];

		}

		template<typename node_t, typename Patch, typename Allocator>
		static BBox3fa createCompressedBVHLeaf(SubdivMesh* mesh, unsigned geomID, unsigned primID, unsigned subpatchID, Patch &patch, PrimRef *prims, const std::vector<Vec2f> &input_uvs, int n_compressed_subd, Allocator& alloc) {
		
			typedef CompressedBVHLeaf<node_t> cLeaf;
			
			cLeaf* leaf = static_cast<cLeaf*>(alloc(sizeof(cLeaf)));
			leaf->nodes = static_cast<node_t*>(alloc(sizeof(node_t) * g_items));
#ifdef WITH_GRID
			leaf->grid = static_cast<Vec3f*>(alloc(sizeof(Vec3f) * (g_edges +1) * (g_edges +1)));
#endif

			array_t<CatmullClarkPatch3fa, 4> subPatches;
			patch.subdivide(subPatches);

			std::vector<Patch>* allPatches = maxSubdivide(patch, LEVEL_Q);
			
			std::vector<Vec3f> finalVertices;

			getDisplacedPositions(mesh, geomID, primID, *allPatches, input_uvs, finalVertices);

			delete allPatches;

			leaf->geom_id = geomID;
			leaf->prim_id = primID;

			size_t zSize = 0;
			size_t tmp = finalVertices.size();
			while (tmp != 0) {
				tmp = tmp >> 1;
				zSize |= tmp;
			}

			std::vector<Vec3f>& v = finalVertices;
			uint32_t i0 = 0;
			uint32_t i1 = 0x55555555 & zSize;
			uint32_t i2 = 0xaaaaaaaa & zSize;
			uint32_t i3 = finalVertices.size() -1;

#ifdef WITH_GRID
			for (uint32_t y = 0; y <= g_edges; ++y) 
				for (uint32_t x = 0; x <= g_edges; ++x) {
					uint32_t idx = 0;
					uint32_t idx2 = y*(g_edges +1)+x;

					if (x == g_edges && y == g_edges)
						idx = EncodeMorton2(x-1,y-1)*4 + 3;
					else if (y == g_edges)
						idx = EncodeMorton2(x, y-1)*4 + 2;
					else if (x == g_edges)
						idx = EncodeMorton2(x-1, y)*4 + 1;
					else
						idx = EncodeMorton2(x, y)*4;

					leaf->grid[idx2] = finalVertices[idx];
				}
#endif

			Vec3fa vx = normalize(v[i1] - v[i0] + v[i3] - v[i2]);
			Vec3fa vy = normalize(v[i2] - v[i0] + v[i3] - v[i1]);
			Vec3fa vz = normalize(cross(vx, vy));
#ifdef ORTHOGONAL
			vy = normalize(cross(vz, vx));
#endif
			const LinearSpace3f world = LinearSpace3f(vx, vy, vz);
			const LinearSpace3f& space = leaf->space = world.inverse();

#ifdef PROJECTION
			const Vec3f v00 = xfmPoint(space, v[i0]);
			const Vec3f v10 = xfmPoint(space, v[i1]);
			const Vec3f v01 = xfmPoint(space, v[i2]);
			const Vec3f v11 = xfmPoint(space, v[i3]);

			std::vector<Vector2f> input;
			input.push_back(Vector2f(v00.x, v00.y));
			input.push_back(Vector2f(v10.x, v10.y));
			input.push_back(Vector2f(v01.x, v01.y));
			input.push_back(Vector2f(v11.x, v11.y));

			std::vector<Vector2f> target;
			target.push_back(Vector2f(-1.f, -1.f));
			target.push_back(Vector2f( 1.f, -1.f));
			target.push_back(Vector2f(-1.f,  1.f));
			target.push_back(Vector2f( 1.f,  1.f));


			bool patchOk = true;
			BBox3f lBox = empty;
			for (size_t i = 0; i < v.size(); i+=4) {
				const Vec3f v00 = xfmPoint(space, v[i+0]);
				const Vec3f v10 = xfmPoint(space, v[i+1]);
				const Vec3f v01 = xfmPoint(space, v[i+2]);
				const Vec3f v11 = xfmPoint(space, v[i+3]);

				if (v00.x > v10.x || v01.x > v11.x || v00.y > v01.y || v10.y > v11.y)
					patchOk = false;

				lBox.extend(v00);
				lBox.extend(v10);
				lBox.extend(v01);
				lBox.extend(v11);
			}

			BBox3f pBox = empty;
			Matrix3f proj;

			if (patchOk == true) {

				proj = ComputeLinearEstimate(input, target);

				for (auto it = v.begin(); it != v.end(); ++it) {
					const Vec3f p = project(xfmPoint(space, *it),proj);

					if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
						p.x < -1.5f || p.x > 1.5f ||
						p.y < -1.5f || p.y > 1.5f) {
						patchOk = false;
						break;
					}

					pBox.extend(p);
				}
			}
			leaf->box = lBox;

			if (patchOk == true) {
				input.clear();
				input.push_back(Vector2f(pBox.lower.x, pBox.lower.y));
				input.push_back(Vector2f(pBox.upper.x, pBox.lower.y));
				input.push_back(Vector2f(pBox.lower.x, pBox.upper.y));
				input.push_back(Vector2f(pBox.upper.x, pBox.upper.y));

				leaf->proj = ComputeLinearEstimate(input, target) * proj;
			}
			else {
				input.clear();
				input.push_back(Vector2f(lBox.lower.x, lBox.lower.y));
				input.push_back(Vector2f(lBox.upper.x, lBox.lower.y));
				input.push_back(Vector2f(lBox.lower.x, lBox.upper.y));
				input.push_back(Vector2f(lBox.upper.x, lBox.upper.y));

				leaf->proj = ComputeLinearEstimate(input, target);
			}
#endif

			buildHierarchy(*leaf, finalVertices);

			leaf->uvs[0] = input_uvs[subpatchID * 4];
			leaf->uvs[1] = input_uvs[subpatchID * 4 + 2] - leaf->uvs[0];


			BBox3f bounds = empty;
			bounds.extend(xfmPoint(world, leaf->box.lower));
			bounds.extend(xfmPoint(world, Vec3f(leaf->box.upper.x, leaf->box.lower.y, leaf->box.lower.z)));
			bounds.extend(xfmPoint(world, Vec3f(leaf->box.lower.x, leaf->box.upper.y, leaf->box.lower.z)));
			bounds.extend(xfmPoint(world, Vec3f(leaf->box.upper.x, leaf->box.upper.y, leaf->box.lower.z)));

			bounds.extend(xfmPoint(world, Vec3f(leaf->box.lower.x, leaf->box.lower.y, leaf->box.upper.z)));
			bounds.extend(xfmPoint(world, Vec3f(leaf->box.upper.x, leaf->box.lower.y, leaf->box.upper.z)));
			bounds.extend(xfmPoint(world, Vec3f(leaf->box.lower.x, leaf->box.upper.y, leaf->box.upper.z)));
			bounds.extend(xfmPoint(world, leaf->box.upper));

			BBox3fa primBounds = BBox3fa(bounds.lower, bounds.upper);

			*prims = PrimRef(primBounds, BVH4::encodeTypedLeaf(leaf, 0));

			return primBounds;
		}

		template<typename node_t = Node<oriented::flavor::non,oriented::quantization::man,oriented::quantization::man2,3,3,2>>
		struct CompressedBVHLeafIntersector1
		{
			typedef CompressedBVHLeaf<node_t> Primitive;

			struct Precalculations { __forceinline Precalculations(const Ray& ray, const void *ptr) {} };

			static __forceinline void intersect(const Precalculations& pre, Ray& ray, const Primitive& prim, Scene* scene, size_t& node) {
				STAT3(normal.trav_prims, 1, 1, 1);


#ifndef PROJECTION
				const Vec3fa org = xfmPoint(static_cast<LinearSpace3fa>(prim.space), ray.org);
				const Vec3fa dir = xfmVector(static_cast<LinearSpace3fa>(prim.space), ray.dir);
#else
				
				const Vec3fa lOrg = xfmPoint(static_cast<LinearSpace3fa>(prim.space), ray.org);
				const Vec3fa lDir = xfmVector(static_cast<LinearSpace3fa>(prim.space), ray.dir);
				
				float near = ray.tnear;
				float far = ray.tfar;
				if (not intersect_clip(prim.box, lOrg, lDir, near, far))
					return;

				Vec3fa org = project(lOrg+lDir*near, prim.proj);
				const Vec3fa tar = project(lOrg+lDir*far, prim.proj);
				Vec3fa dir = tar - org;
				float zFactor; 
				bool special = false;
				float tfar = 0.f;
				const Vec3fa test = abs(dir);

				if (unlikely(test.x < g_epsilon && test.y < g_epsilon && test.z < g_epsilon)) {
					dir.z = copysign(1.f, lDir.z);
					org.z -= dir.z;
					zFactor = FLT_MAX;
					tfar = FLT_MAX;
				}
				else if (unlikely(test.z < g_epsilon)) {
					special = true;
				  	tfar = length(dir);
					dir = normalize(dir);
				}
				else {
					dir = normalize(dir);
					zFactor = lDir.z / dir.z;
					tfar = (ray.tfar - near) * zFactor;
				}

#endif

#ifndef WITH_GRID
#ifndef EARLY_OUT
				float dist[20];
				dist[0] = ray.tfar;
#endif
#endif
				size_t stack[20];
				BBox3fa boxStack[20];
				int sp = 0;

				stack[0] = 0;
#ifndef PROJECTION
				boxStack[0] = BBox3fa(prim.box.lower, prim.box.upper);
#else
				boxStack[0] = BBox3fa(Vec3fa(-1.f, -1.f, prim.box.lower.z),
									 Vec3fa( 1.f,  1.f, prim.box.upper.z));
#endif

				/*! load the ray into SIMD registers */
				const Vec3fa ray_rdir = rcp_safe(dir);
				const Vec3fa ray_org_rdir = org*ray_rdir;
				const Vec3f4 simd_org(org.x,org.y,org.z);
				const Vec3f4 simd_dir(dir.x,dir.y,dir.z);
				const Vec3f4 simd_rdir(ray_rdir.x,ray_rdir.y,ray_rdir.z);
				const Vec3f4 simd_org_rdir(ray_org_rdir.x,ray_org_rdir.y,ray_org_rdir.z);
#ifndef PROJECTION
				const float4 ray_near(ray.tnear);
				float4 ray_far(ray.tfar);
#else	
				const float4 ray_near(0.f);
				float4 ray_far(tfar);
#endif


				/*! offsets to select the side that becomes the lower or upper bound */
				const size_t nearX = ray_rdir.x >= 0.0f ? 0*sizeof(float4) : 1*sizeof(float4);
				const size_t nearY = ray_rdir.y >= 0.0f ? 2*sizeof(float4) : 3*sizeof(float4);
				const size_t nearZ = ray_rdir.z >= 0.0f ? 4*sizeof(float4) : 5*sizeof(float4);
				float4 tNear;

				while (sp >= 0) {
#ifndef WITH_GRID
#ifndef EARLY_OUT
					const float is = dist[sp];
#endif
#endif
					const size_t& curr = stack[sp];
					const BBox3fa& pBox = boxStack[sp--];

					if (curr > g_items - 1) {

#ifdef WITH_GRID
						const unsigned int idx = curr - g_items;
						if (intersect_triangles(idx, prim.grid, ray)) {
							ray.geomID = prim.geom_id;
							ray.primID = prim.prim_id;
							ray.u = prim.uvs[0].x + ray.u * prim.uvs[1].x;
							ray.v = prim.uvs[0].y + ray.v * prim.uvs[1].y;
#ifdef EARLY_OUT
							return;
#endif
						}
#else

#ifdef EARLY_OUT
						const float is = reduce_min(tNear);
#endif


#ifndef PROJECTION
						if (is < ray.tfar) {
#else
						if (is < tfar) {
#endif
							const unsigned int idx = curr - g_items;

							const float x = (((org.x + dir.x * is) - pBox.lower.x) / (pBox.upper.x - pBox.lower.x) +
										 	(float)DecodeMorton2X(idx)) * g_rcpEdges;

							const float y = (((org.y + dir.y * is) - pBox.lower.y) / (pBox.upper.y - pBox.lower.y) +
											 (float)DecodeMorton2Y(idx)) * g_rcpEdges;

							ray.u = prim.uvs[0].x + x * prim.uvs[1].x;
							ray.v = prim.uvs[0].y + y * prim.uvs[1].y;

							ray.geomID = prim.geom_id;
							ray.primID = prim.prim_id;
							
#ifndef PROJECTION
							ray.tfar = is;
#else
							if (unlikely(special == true)) {
								const Vec3fa p = project(org + dir * is, prim.proj.inverse());
								ray.tfar = length(p - lOrg);
							}
							else
								ray.tfar = is / zFactor + near;


#ifdef EARLY_OUT
							return;
#else
							tfar = is;
#endif
#endif
							
						}
#endif
						continue;
					}

					const unsigned int next = curr * 4 + 1;
					const node_t& cNode = prim.nodes[curr];
					size_t mask;
					const BVH4::Node eNode = cNode.getNode(pBox);

					mask = isa::intersect_node<true>(&eNode,nearX,nearY,nearZ,simd_org,simd_rdir,simd_org_rdir,ray_near,ray_far,tNear);

					// nothing hit
					if (unlikely(mask == 0))
						continue;

					// one child hit
					size_t r = __bscf(mask);
					if (likely(mask==0)) {
						boxStack[++sp] = eNode.bounds(r);
						stack[sp] = next + r;
#ifndef WITH_GRID
#ifndef EARLY_OUT
						dist[sp] = tNear[r];
#endif
#endif
						continue;
					}

					// two children hit
					const size_t c0 = r;
					const unsigned int d0 = ((unsigned int*)&tNear)[r];
					r = __bscf(mask);
					const unsigned int d1 = ((unsigned int*)&tNear)[r];
					if (likely(mask==0)) {
						if (d0 > d1) {
							boxStack[++sp] = eNode.bounds(c0); stack[sp] = next + c0;
#ifndef WITH_GRID
#ifndef EARLY_OUT
							dist[sp] = tNear[c0];
#endif
#endif
							boxStack[++sp] = eNode.bounds( r); stack[sp] = next +  r;
#ifndef WITH_GRID
#ifndef EARLY_OUT
							dist[sp] = tNear[r];
#endif
#endif
						}
						else {
							boxStack[++sp] = eNode.bounds( r); stack[sp] = next +  r;
#ifndef WITH_GRID
#ifndef EARLY_OUT
							dist[sp] = tNear[r];
#endif
#endif
							boxStack[++sp] = eNode.bounds(c0); stack[sp] = next + c0;
#ifndef WITH_GRID
#ifndef EARLY_OUT
							dist[sp] = tNear[c0];
#endif
#endif
						}
						continue;
					}

					// three children hit
					const size_t c1 = r;
					r = __bscf(mask);
					const unsigned int d2 = ((unsigned int*)&tNear)[r];
					if (likely(mask==0)) {
						const size_t shift0 = sp + 1 + (d0 <= d1) + (d0 <= d2);
						const size_t shift1 = sp + 1 + (d1 <  d0) + (d1 <= d2);
						const size_t shift2 = sp + 1 + (d2 <  d0) + (d2 <  d1);

						boxStack[shift0] = eNode.bounds(c0);
						boxStack[shift1] = eNode.bounds(c1);
						boxStack[shift2] = eNode.bounds(r);

						stack[shift0] = next + c0;
						stack[shift1] = next + c1;
						stack[shift2] = next + r;

#ifndef WITH_GRID
#ifndef EARLY_OUT
						dist[shift0] = tNear[c0];
						dist[shift1] = tNear[c1];
						dist[shift2] = tNear[r];
#endif
#endif

						sp += 3;
						continue;
					}

					// four children hit
					const size_t c2 = r;
					r = __bscf(mask);
					const unsigned int d3 = ((unsigned int*)&tNear)[r];

					const size_t shift0 = sp + 1 + (d0 <= d1) + (d0 <= d2) + (d0 <= d3);
					const size_t shift1 = sp + 1 + (d1 <  d0) + (d1 <= d2) + (d1 <= d3);
					const size_t shift2 = sp + 1 + (d2 <  d0) + (d2 <  d1) + (d2 <= d3);
					const size_t shift3 = sp + 1 + (d3 <  d0) + (d3 <  d1) + (d3 <  d2);

					boxStack[shift0] = eNode.bounds(c0);
					boxStack[shift1] = eNode.bounds(c1);
					boxStack[shift2] = eNode.bounds(c2);
					boxStack[shift3] = eNode.bounds(r);

					stack[shift0] = next + c0;
					stack[shift1] = next + c1;
					stack[shift2] = next + c2;
					stack[shift3] = next + r;
#ifndef WITH_GRID
#ifndef EARLY_OUT
					dist[shift0] = tNear[c0];
					dist[shift1] = tNear[c1];
					dist[shift2] = tNear[c2];
					dist[shift3] = tNear[r];
#endif
#endif

					sp += 4;
				}
			}

			static __forceinline void intersect(const Precalculations& pre, Ray& ray, const Primitive* prim, size_t ty, Scene* scene, size_t& node) {
				intersect(pre, ray, prim[0], scene, node);
			}
			
			static __forceinline bool occluded(const Precalculations& pre, Ray& ray, const Primitive& prim, Scene* scene, size_t& node) {
				STAT3(shadow.trav_prims, 1, 1, 1);
				
				//TODO more optimization
				ray.tnear += 0.1f;
				intersect(pre, ray, prim, scene, node);

				return ray;
			}
			static __forceinline bool occluded(const Precalculations& pre, Ray& ray, const Primitive* prim, size_t ty, Scene* scene, size_t& node) {
				return occluded(pre, ray, prim[0], scene, node);
			}
		};
	}
}

// vim: set foldmethod=marker :
