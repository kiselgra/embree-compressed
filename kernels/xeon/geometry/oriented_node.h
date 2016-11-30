#pragma once

#include <vector>
#include <cmath>
#include <iostream>

#include "../bvh4/bvh4.h"
#include "../../../common/math/bbox.h"
#include "oriented_table.h"

namespace embree {
	namespace oriented {

		struct flavor {
			enum {com, mid, non, ref };
		};

		class NodeStorageBase {
		public:
			inline void setX1(unsigned char val) {}
			inline void setX2(unsigned char val) {}
			inline void setX3(unsigned char val) {}
			inline void setX4(unsigned char val) {}

			inline void setY1(unsigned char val) {}
			inline void setY2(unsigned char val) {}
			inline void setY3(unsigned char val) {}
			inline void setY4(unsigned char val) {}

			inline void setZ1(unsigned char val) {}
			inline void setZ2(unsigned char val) {}
			inline void setZ3(unsigned char val) {}
			inline void setZ4(unsigned char val) {}

			inline void setMinZ(unsigned char val) {}
			inline void setMaxZ(unsigned char val) {}

			inline unsigned char getX1() const { return 0; }
			inline unsigned char getX2() const { return 0; }
			inline unsigned char getX3() const { return 0; }
			inline unsigned char getX4() const { return 0; }

			inline unsigned char getY1() const { return 0; }
			inline unsigned char getY2() const { return 0; }
			inline unsigned char getY3() const { return 0; }
			inline unsigned char getY4() const { return 0; }

			inline unsigned char getZ1() const { return 0; }
			inline unsigned char getZ2() const { return 0; }
			inline unsigned char getZ3() const { return 0; }
			inline unsigned char getZ4() const { return 0; }

			inline unsigned char getMinZ() const { return 0; }
			inline unsigned char getMaxZ() const { return 0; }

			inline void setMinXb1(unsigned char val) {}
			inline void setMaxXb1(unsigned char val) {}
			inline void setMinYb1(unsigned char val) {}
			inline void setMaxYb1(unsigned char val) {}
			inline void setMinZb1(unsigned char val) {}
			inline void setMaxZb1(unsigned char val) {}

			inline unsigned char getMinXb1() const { return 0; }
			inline unsigned char getMaxXb1() const { return 0; }
			inline unsigned char getMinYb1() const { return 0; }
			inline unsigned char getMaxYb1() const { return 0; }
			inline unsigned char getMinZb1() const { return 0; }
			inline unsigned char getMaxZb1() const { return 0; }

			inline void setMinXb2(unsigned char val) {}
			inline void setMaxXb2(unsigned char val) {}
			inline void setMinYb2(unsigned char val) {}
			inline void setMaxYb2(unsigned char val) {}
			inline void setMinZb2(unsigned char val) {}
			inline void setMaxZb2(unsigned char val) {}

			inline unsigned char getMinXb2() const { return 0; }
			inline unsigned char getMaxXb2() const { return 0; }
			inline unsigned char getMinYb2() const { return 0; }
			inline unsigned char getMaxYb2() const { return 0; }
			inline unsigned char getMinZb2() const { return 0; }
			inline unsigned char getMaxZb2() const { return 0; }

			inline void setMinXb3(unsigned char val) {}
			inline void setMaxXb3(unsigned char val) {}
			inline void setMinYb3(unsigned char val) {}
			inline void setMaxYb3(unsigned char val) {}
			inline void setMinZb3(unsigned char val) {}
			inline void setMaxZb3(unsigned char val) {}

			inline unsigned char getMinXb3() const { return 0; }
			inline unsigned char getMaxXb3() const { return 0; }
			inline unsigned char getMinYb3() const { return 0; }
			inline unsigned char getMaxYb3() const { return 0; }
			inline unsigned char getMinZb3() const { return 0; }
			inline unsigned char getMaxZb3() const { return 0; }

			inline void setMinXb4(unsigned char val) {}
			inline void setMaxXb4(unsigned char val) {}
			inline void setMinYb4(unsigned char val) {}
			inline void setMaxYb4(unsigned char val) {}
			inline void setMinZb4(unsigned char val) {}
			inline void setMaxZb4(unsigned char val) {}

			inline unsigned char getMinXb4() const { return 0; }
			inline unsigned char getMaxXb4() const { return 0; }
			inline unsigned char getMinYb4() const { return 0; }
			inline unsigned char getMaxYb4() const { return 0; }
			inline unsigned char getMinZb4() const { return 0; }
			inline unsigned char getMaxZb4() const { return 0; }
		};

		template <int flavor, int xBits, int yBits, int zBits>
			class NodeStorage : public NodeStorageBase {};

		template<>
			class NodeStorage<flavor::mid, 3, 3, 2> : public NodeStorageBase {
			private:
				unsigned char xz;
				unsigned char yz;
			public:
				inline void setX2(unsigned char min) { xz = (xz & 0b00011111) | min << 5; }
				inline void setX3(unsigned char max) { xz = (xz & 0b11100011) | max << 2; }
				inline void setY2(unsigned char min) { yz = (yz & 0b00011111) | min << 5; }
				inline void setY3(unsigned char max) { yz = (yz & 0b11100011) | max << 2; }
				inline void setMinZ(unsigned char min) { xz = (xz & 0b11111100) | min; }
				inline void setMaxZ(unsigned char max) { yz = (yz & 0b11111100) | max; }

				inline unsigned char getX2() const { return (xz & 0b11100000) >> 5; }
				inline unsigned char getX3() const { return (xz & 0b00011100) >> 2; }
				inline unsigned char getY2() const { return (yz & 0b11100000) >> 5; }
				inline unsigned char getY3() const { return (yz & 0b00011100) >> 2; }
				inline unsigned char getMinZ() const { return (xz & 0b00000011); }
				inline unsigned char getMaxZ() const { return (yz & 0b00000011); }
			};

		template<>
			class NodeStorage<flavor::com, 3, 3, 2> : public NodeStorageBase {
			public:
				unsigned char xz; // + z max
				unsigned char x;  
				unsigned char yz; // + z max
				unsigned char y;  
			public:
				inline void setX1(unsigned char val) { xz = ( xz & 0b00011111) | val << 5; }
				inline void setX2(unsigned char val) { xz = ( xz & 0b11100011) | val << 2; }
				inline void setX3(unsigned char val) { x  = ( x  & 0b00011111) | val << 5; }
				inline void setX4(unsigned char val) { x  = ( x  & 0b11100011) | val << 2; }

				inline void setY1(unsigned char val) { yz = ( yz & 0b00011111) | val << 5; }
				inline void setY2(unsigned char val) { yz = ( yz & 0b11100011) | val << 2; }
				inline void setY3(unsigned char val) { y  = ( y  & 0b00011111) | val << 5; }
				inline void setY4(unsigned char val) { y  = ( y  & 0b11100011) | val << 2; }

				inline void setMinZ(unsigned char val) { xz = (xz & 0b11111100) | val; }
				inline void setMaxZ(unsigned char val) { yz = (yz & 0b11111100) | val; }

				inline unsigned char getX1() const { return (xz & 0b11100000) >> 5; }
				inline unsigned char getX2() const { return (xz & 0b00011100) >> 2; }
				inline unsigned char getX3() const { return (x  & 0b11100000) >> 5; }
				inline unsigned char getX4() const { return (x  & 0b00011100) >> 2; }

				inline unsigned char getY1() const { return (yz & 0b11100000) >> 5; }
				inline unsigned char getY2() const { return (yz & 0b00011100) >> 2; }
				inline unsigned char getY3() const { return (y  & 0b11100000) >> 5; }
				inline unsigned char getY4() const { return (y  & 0b00011100) >> 2; }

				inline unsigned char getMinZ() const { return (xz & 0b00000011); }
				inline unsigned char getMaxZ() const { return (yz & 0b00000011); }

			};

		template<>
			class NodeStorage<flavor::non, 3, 3, 2> : public NodeStorageBase {
			private:
				unsigned char b1xz;
				unsigned char b1yz;

				unsigned char b2xz;
				unsigned char b2yz;

				unsigned char b3xz;
				unsigned char b3yz;

				unsigned char b4xz;
				unsigned char b4yz;
			public:

				inline void setMinXb1(unsigned char val) { b1xz = (b1xz & 0b00011111) | val << 5;}
				inline void setMaxXb1(unsigned char val) { b1xz = (b1xz & 0b11100011) | val << 2;}
				inline void setMinYb1(unsigned char val) { b1yz = (b1yz & 0b00011111) | val << 5;}
				inline void setMaxYb1(unsigned char val) { b1yz = (b1yz & 0b11100011) | val << 2;}
				inline void setMinZb1(unsigned char val) { b1xz = (b1xz & 0b11111100) | val;}
				inline void setMaxZb1(unsigned char val) { b1yz = (b1yz & 0b11111100) | val;}

				inline unsigned char getMinXb1() const { return (b1xz & 0b11100000) >> 5; }
				inline unsigned char getMaxXb1() const { return (b1xz & 0b00011100) >> 2; }
				inline unsigned char getMinYb1() const { return (b1yz & 0b11100000) >> 5; }
				inline unsigned char getMaxYb1() const { return (b1yz & 0b00011100) >> 2; }
				inline unsigned char getMinZb1() const { return (b1xz & 0b00000011); }
				inline unsigned char getMaxZb1() const { return (b1yz & 0b00000011); }

				inline void setMinXb2(unsigned char val) { b2xz = (b2xz & 0b00011111) | val << 5;}
				inline void setMaxXb2(unsigned char val) { b2xz = (b2xz & 0b11100011) | val << 2;}
				inline void setMinYb2(unsigned char val) { b2yz = (b2yz & 0b00011111) | val << 5;}
				inline void setMaxYb2(unsigned char val) { b2yz = (b2yz & 0b11100011) | val << 2;}
				inline void setMinZb2(unsigned char val) { b2xz = (b2xz & 0b11111100) | val;}
				inline void setMaxZb2(unsigned char val) { b2yz = (b2yz & 0b11111100) | val;}

				inline unsigned char getMinXb2() const { return (b2xz & 0b11100000) >> 5; }
				inline unsigned char getMaxXb2() const { return (b2xz & 0b00011100) >> 2; }
				inline unsigned char getMinYb2() const { return (b2yz & 0b11100000) >> 5; }
				inline unsigned char getMaxYb2() const { return (b2yz & 0b00011100) >> 2; }
				inline unsigned char getMinZb2() const { return (b2xz & 0b00000011); }
				inline unsigned char getMaxZb2() const { return (b2yz & 0b00000011); }

				inline void setMinXb3(unsigned char val) { b3xz = (b3xz & 0b00011111) | val << 5;}
				inline void setMaxXb3(unsigned char val) { b3xz = (b3xz & 0b11100011) | val << 2;}
				inline void setMinYb3(unsigned char val) { b3yz = (b3yz & 0b00011111) | val << 5;}
				inline void setMaxYb3(unsigned char val) { b3yz = (b3yz & 0b11100011) | val << 2;}
				inline void setMinZb3(unsigned char val) { b3xz = (b3xz & 0b11111100) | val;}
				inline void setMaxZb3(unsigned char val) { b3yz = (b3yz & 0b11111100) | val;}

				inline unsigned char getMinXb3() const { return (b3xz & 0b11100000) >> 5; }
				inline unsigned char getMaxXb3() const { return (b3xz & 0b00011100) >> 2; }
				inline unsigned char getMinYb3() const { return (b3yz & 0b11100000) >> 5; }
				inline unsigned char getMaxYb3() const { return (b3yz & 0b00011100) >> 2; }
				inline unsigned char getMinZb3() const { return (b3xz & 0b00000011); }
				inline unsigned char getMaxZb3() const { return (b3yz & 0b00000011); }

				inline void setMinXb4(unsigned char val) { b4xz = (b4xz & 0b00011111) | val << 5;}
				inline void setMaxXb4(unsigned char val) { b4xz = (b4xz & 0b11100011) | val << 2;}
				inline void setMinYb4(unsigned char val) { b4yz = (b4yz & 0b00011111) | val << 5;}
				inline void setMaxYb4(unsigned char val) { b4yz = (b4yz & 0b11100011) | val << 2;}
				inline void setMinZb4(unsigned char val) { b4xz = (b4xz & 0b11111100) | val;}
				inline void setMaxZb4(unsigned char val) { b4yz = (b4yz & 0b11111100) | val;}

				inline unsigned char getMinXb4() const { return (b4xz & 0b11100000) >> 5; }
				inline unsigned char getMaxXb4() const { return (b4xz & 0b00011100) >> 2; }
				inline unsigned char getMinYb4() const { return (b4yz & 0b11100000) >> 5; }
				inline unsigned char getMaxYb4() const { return (b4yz & 0b00011100) >> 2; }
				inline unsigned char getMinZb4() const { return (b4xz & 0b00000011); }
				inline unsigned char getMaxZb4() const { return (b4yz & 0b00000011); }
			};

		template<>
			class NodeStorage<flavor::ref, 32, 32, 32> : public NodeStorageBase {
			private:
				float boxes[4][6];
			public:

				inline void setMinX(float val, int i) { boxes[i][0] = val; }
				inline void setMaxX(float val, int i) { boxes[i][1] = val; }
				inline void setMinY(float val, int i) { boxes[i][2] = val; }
				inline void setMaxY(float val, int i) { boxes[i][3] = val; }
				inline void setMinZ(float val, int i) { boxes[i][4] = val; }
				inline void setMaxZ(float val, int i) { boxes[i][5] = val; }

				inline float getMinX(int i) const { return boxes[i][0]; }
				inline float getMaxX(int i) const { return boxes[i][1]; }
				inline float getMinY(int i) const { return boxes[i][2]; }
				inline float getMaxY(int i) const { return boxes[i][3]; }
				inline float getMinZ(int i) const { return boxes[i][4]; }
				inline float getMaxZ(int i) const { return boxes[i][5]; }
			};

		class BaseNode {
		public:
		};



		template <int flavor, int quant, int quant2, int xBits, int yBits, int zBits>
			class Node : public BaseNode {
			public:
				typedef NodeStorage<flavor, xBits, yBits, zBits> nodeStorage_t;
				NodeStorage<flavor, xBits, yBits, zBits> xyz;
				static LookupTable<quant, xBits> table1;
				static LookupTable<quant2, yBits> table2;
				static LookupTable<quantization::uni, zBits> table3;

				void setAABB(const BBox3fa& parentBox,
							 const BBox3fa& box00,
							 const BBox3fa& box10,
							 const BBox3fa& box01,
							 const BBox3fa& box11) {

					float x1, x2, x3, x4;
					float y1, y2, y3, y4;
					float z1, z2;

					double xF = 1.0 / (parentBox.upper.x - parentBox.lower.x);
					double yF = 1.0 / (parentBox.upper.y - parentBox.lower.y);
					double zF = 1.0 / (parentBox.upper.z - parentBox.lower.z);

					if (!std::isfinite(xF)) xF = FLT_MIN;
					if (!std::isfinite(yF)) yF = FLT_MIN;
					if (!std::isfinite(zF)) zF = FLT_MIN;

					x1 = fmin(box00.lower.x, box01.lower.x);
					x2 = fmin(box10.lower.x, box11.lower.x);
					x3 = fmax(box00.upper.x, box01.upper.x);
					x4 = fmax(box10.upper.x, box11.upper.x);

					y1 = fmin(box00.lower.y, box10.lower.y);
					y2 = fmin(box01.lower.y, box11.lower.y);
					y3 = fmax(box00.upper.y, box10.upper.y);
					y4 = fmax(box01.upper.y, box11.upper.y);

					z1 = fmin(fmin(box00.lower.z, box10.lower.z), fmin(box01.lower.z, box11.lower.z));
					z2 = fmax(fmax(box00.upper.z, box10.upper.z), fmax(box01.upper.z, box11.upper.z));

					xyz.setX1(table1.lookUpIdx(((x1 - parentBox.lower.x) * xF)));
					xyz.setX2(table2.lookUpIdx(((x2 - parentBox.lower.x) * xF)));
					xyz.setX3(table2.lookUpIdx(((parentBox.upper.x - x3) * xF)));
					xyz.setX4(table1.lookUpIdx(((parentBox.upper.x - x4) * xF)));

					xyz.setY1(table1.lookUpIdx(((y1 - parentBox.lower.y) * yF)));
					xyz.setY2(table2.lookUpIdx(((y2 - parentBox.lower.y) * yF)));
					xyz.setY3(table2.lookUpIdx(((parentBox.upper.y - y3) * yF)));
					xyz.setY4(table1.lookUpIdx(((parentBox.upper.y - y4) * yF)));

					xyz.setMinZ(table3.lookUpIdx((z1 - parentBox.lower.z) * zF));
					xyz.setMaxZ(table3.lookUpIdx((parentBox.upper.z - z2) * zF));

				}

				void getAABB(BBox3fa& retBox, const BBox3fa& parentBox, const int loc) const {
					Vec3fa dim = parentBox.upper - parentBox.lower;
					Vec3fa min = Vec3fa(0.f, 0.f, table3.lookUp(xyz.getMinZ()));
					Vec3fa max = Vec3fa(0.f, 0.f, 1.f - table3.lookUp(xyz.getMaxZ()));

					switch(loc) {
					case 0:
						min.x = table1.lookUp(xyz.getX1());
						min.y = table1.lookUp(xyz.getY1());
						max.x = 1.f - table2.lookUp(xyz.getX3());
						max.y = 1.f - table2.lookUp(xyz.getY3());
						break;
					case 1:
						min.x = table2.lookUp(xyz.getX2());
						min.y = table1.lookUp(xyz.getY1());
						max.x = 1.f - table1.lookUp(xyz.getX4());
						max.y = 1.f - table2.lookUp(xyz.getY3());
						break;
					case 2:
						min.x = table1.lookUp(xyz.getX1());
						min.y = table2.lookUp(xyz.getY2());
						max.x = 1.f - table2.lookUp(xyz.getX3());
						max.y = 1.f - table1.lookUp(xyz.getY4());
						break;
					case 3:
						min.x = table2.lookUp(xyz.getX2());
						min.y = table2.lookUp(xyz.getY2());
						max.x = 1.f - table1.lookUp(xyz.getX4());
						max.y = 1.f - table1.lookUp(xyz.getY4());
						break;
					}

					retBox.lower = min * dim + parentBox.lower;
					retBox.upper = max * dim + parentBox.lower;
				}

				BVH4::Node getNode(const BBox3fa& parentBox) const {
					const float dimX = parentBox.upper.x - parentBox.lower.x;
					const float dimY = parentBox.upper.y - parentBox.lower.y;
					const float dimZ = parentBox.upper.z - parentBox.lower.z;
					BVH4::Node node;

					node.lower_x[0] = node.lower_x[2] = table1.lookUp(xyz.getX1()) * dimX + parentBox.lower.x;
					node.lower_x[1] = node.lower_x[3] = table2.lookUp(xyz.getX2()) * dimX + parentBox.lower.x;

					node.upper_x[0] = node.upper_x[2] = (1.f - table2.lookUp(xyz.getX3())) * dimX + parentBox.lower.x;
					node.upper_x[1] = node.upper_x[3] = (1.f - table1.lookUp(xyz.getX4())) * dimX + parentBox.lower.x;

					node.lower_y[0] = node.lower_y[1] = table1.lookUp(xyz.getY1()) * dimY + parentBox.lower.y;
					node.lower_y[2] = node.lower_y[3] = table2.lookUp(xyz.getY2()) * dimY + parentBox.lower.y;

					node.upper_y[0] = node.upper_y[1] = (1.f - table2.lookUp(xyz.getY3())) * dimY + parentBox.lower.y;
					node.upper_y[2] = node.upper_y[3] = (1.f - table1.lookUp(xyz.getY4())) * dimY + parentBox.lower.y;

					node.lower_z[0] = node.lower_z[1] = node.lower_z[2] = node.lower_z[3] = table3.lookUp(xyz.getMinZ()) * dimZ + parentBox.lower.z;
					node.upper_z[0] = node.upper_z[1] = node.upper_z[2] = node.upper_z[3] = (1.f - table3.lookUp(xyz.getMaxZ())) * dimZ + parentBox.lower.z;

					return node;
				}

			};

		// uncompressed non
		template <int quant, int quant2, int xBits, int yBits, int zBits>
			class Node<flavor::non, quant, quant2, xBits, yBits, zBits> : public BaseNode {
			public:
				typedef NodeStorage<flavor::non, xBits, yBits, zBits> nodeStorage_t;
				NodeStorage<flavor::non, xBits, yBits, zBits> xyz;
				static LookupTable<quant, xBits> table1;
				static LookupTable<quant2, yBits> table2;
				static LookupTable<quantization::uni, zBits> table3;

				void setAABB(const BBox3fa& parentBox,
							 const BBox3fa& box00,
							 const BBox3fa& box10,
							 const BBox3fa& box01,
							 const BBox3fa& box11) {

					double xF = 1.0 / (parentBox.upper.x - parentBox.lower.x);
					double yF = 1.0 / (parentBox.upper.y - parentBox.lower.y);
					double zF = 1.0 / (parentBox.upper.z - parentBox.lower.z);

					if (!std::isfinite(xF)) xF = FLT_MIN;
					if (!std::isfinite(yF)) yF = FLT_MIN;
					if (!std::isfinite(zF)) zF = FLT_MIN;

					// bo00
					xyz.setMinXb1(table1.lookUpIdx((box00.lower.x - parentBox.lower.x) * xF));
					xyz.setMinYb1(table1.lookUpIdx((box00.lower.y - parentBox.lower.y) * yF));
					xyz.setMinZb1(table3.lookUpIdx((box00.lower.z - parentBox.lower.z) * zF));

					xyz.setMaxXb1(table2.lookUpIdx((parentBox.upper.x - box00.upper.x) * xF));
					xyz.setMaxYb1(table2.lookUpIdx((parentBox.upper.y - box00.upper.y) * yF));
					xyz.setMaxZb1(table3.lookUpIdx((parentBox.upper.z - box00.upper.z) * zF));

					// bo10
					xyz.setMinXb2(table2.lookUpIdx((box10.lower.x - parentBox.lower.x) * xF));
					xyz.setMinYb2(table1.lookUpIdx((box10.lower.y - parentBox.lower.y) * yF));
					xyz.setMinZb2(table3.lookUpIdx((box10.lower.z - parentBox.lower.z) * zF));

					xyz.setMaxXb2(table1.lookUpIdx((parentBox.upper.x - box10.upper.x) * xF));
					xyz.setMaxYb2(table2.lookUpIdx((parentBox.upper.y - box10.upper.y) * yF));
					xyz.setMaxZb2(table3.lookUpIdx((parentBox.upper.z - box10.upper.z) * zF));

					// bo01
					xyz.setMinXb3(table1.lookUpIdx((box01.lower.x - parentBox.lower.x) * xF));
					xyz.setMinYb3(table2.lookUpIdx((box01.lower.y - parentBox.lower.y) * yF));
					xyz.setMinZb3(table3.lookUpIdx((box01.lower.z - parentBox.lower.z) * zF));

					xyz.setMaxXb3(table2.lookUpIdx((parentBox.upper.x - box01.upper.x) * xF));
					xyz.setMaxYb3(table1.lookUpIdx((parentBox.upper.y - box01.upper.y) * yF));
					xyz.setMaxZb3(table3.lookUpIdx((parentBox.upper.z - box01.upper.z) * zF));

					// bo11
					xyz.setMinXb4(table2.lookUpIdx((box11.lower.x - parentBox.lower.x) * xF));
					xyz.setMinYb4(table2.lookUpIdx((box11.lower.y - parentBox.lower.y) * yF));
					xyz.setMinZb4(table3.lookUpIdx((box11.lower.z - parentBox.lower.z) * zF));

					xyz.setMaxXb4(table1.lookUpIdx((parentBox.upper.x - box11.upper.x) * xF));
					xyz.setMaxYb4(table1.lookUpIdx((parentBox.upper.y - box11.upper.y) * yF));
					xyz.setMaxZb4(table3.lookUpIdx((parentBox.upper.z - box11.upper.z) * zF));


				}

				void getAABB(BBox3fa& retBox, const BBox3fa& parentBox, const int loc) const {
					Vec3fa dim = parentBox.upper - parentBox.lower;
					Vec3fa min = Vec3fa(0.f, 0.f, 0.f);
					Vec3fa max = Vec3fa(0.f, 0.f, 0.f);

					switch(loc) {
					case 0:
						min.x = table1.lookUp(xyz.getMinXb1());
						min.y = table1.lookUp(xyz.getMinYb1());
						min.z = table3.lookUp(xyz.getMinZb1());
						max.x = 1.f - table2.lookUp(xyz.getMaxXb1());
						max.y = 1.f - table2.lookUp(xyz.getMaxYb1());
						max.z = 1.f - table3.lookUp(xyz.getMaxZb1());
						break;
					case 1:
						min.x = table2.lookUp(xyz.getMinXb2());
						min.y = table1.lookUp(xyz.getMinYb2());
						min.z = table3.lookUp(xyz.getMinZb2());
						max.x = 1.f - table1.lookUp(xyz.getMaxXb2());
						max.y = 1.f - table2.lookUp(xyz.getMaxYb2());
						max.z = 1.f - table3.lookUp(xyz.getMaxZb2());
						break;
					case 2:
						min.x = table1.lookUp(xyz.getMinXb3());
						min.y = table2.lookUp(xyz.getMinYb3());
						min.z = table3.lookUp(xyz.getMinZb3());
						max.x = 1.f - table2.lookUp(xyz.getMaxXb3());
						max.y = 1.f - table1.lookUp(xyz.getMaxYb3());
						max.z = 1.f - table3.lookUp(xyz.getMaxZb3());
						break;
					case 3:
						min.x = table2.lookUp(xyz.getMinXb4());
						min.y = table2.lookUp(xyz.getMinYb4());
						min.z = table3.lookUp(xyz.getMinZb4());
						max.x = 1.f - table1.lookUp(xyz.getMaxXb4());
						max.y = 1.f - table1.lookUp(xyz.getMaxYb4());
						max.z = 1.f - table3.lookUp(xyz.getMaxZb4());
						break;
					}
					retBox.lower = min * dim + parentBox.lower;
					retBox.upper = max * dim + parentBox.lower;

				}

				BVH4::Node getNode(const BBox3fa& parentBox) const {
					const float dimX = parentBox.upper.x - parentBox.lower.x;
					const float dimY = parentBox.upper.y - parentBox.lower.y;
					const float dimZ = parentBox.upper.z - parentBox.lower.z;
					BVH4::Node node;

					node.lower_x[0] = table1.lookUp(xyz.getMinXb1()) * dimX + parentBox.lower.x;
					node.lower_y[0] = table1.lookUp(xyz.getMinYb1()) * dimY + parentBox.lower.y;
					node.lower_z[0] = table3.lookUp(xyz.getMinZb1()) * dimZ + parentBox.lower.z;
					node.upper_x[0] = (1.f - table2.lookUp(xyz.getMaxXb1())) * dimX + parentBox.lower.x;
					node.upper_y[0] = (1.f - table2.lookUp(xyz.getMaxYb1())) * dimY + parentBox.lower.y;
					node.upper_z[0] = (1.f - table3.lookUp(xyz.getMaxZb1())) * dimZ + parentBox.lower.z;

					node.lower_x[1] = table2.lookUp(xyz.getMinXb2()) * dimX + parentBox.lower.x;
					node.lower_y[1] = table1.lookUp(xyz.getMinYb2()) * dimY + parentBox.lower.y;
					node.lower_z[1] = table3.lookUp(xyz.getMinZb2()) * dimZ + parentBox.lower.z;
					node.upper_x[1] = (1.f - table1.lookUp(xyz.getMaxXb2())) * dimX + parentBox.lower.x;
					node.upper_y[1] = (1.f - table2.lookUp(xyz.getMaxYb2())) * dimY + parentBox.lower.y;
					node.upper_z[1] = (1.f - table3.lookUp(xyz.getMaxZb2())) * dimZ + parentBox.lower.z;

					node.lower_x[2] = table1.lookUp(xyz.getMinXb3()) * dimX + parentBox.lower.x;
					node.lower_y[2] = table2.lookUp(xyz.getMinYb3()) * dimY + parentBox.lower.y;
					node.lower_z[2] = table3.lookUp(xyz.getMinZb3()) * dimZ + parentBox.lower.z;
					node.upper_x[2] = (1.f - table2.lookUp(xyz.getMaxXb3())) * dimX + parentBox.lower.x;
					node.upper_y[2] = (1.f - table1.lookUp(xyz.getMaxYb3())) * dimY + parentBox.lower.y;
					node.upper_z[2] = (1.f - table3.lookUp(xyz.getMaxZb3())) * dimZ + parentBox.lower.z;

					node.lower_x[3] = table2.lookUp(xyz.getMinXb4()) * dimX + parentBox.lower.x;
					node.lower_y[3] = table2.lookUp(xyz.getMinYb4()) * dimY + parentBox.lower.y;
					node.lower_z[3] = table3.lookUp(xyz.getMinZb4()) * dimZ + parentBox.lower.z;
					node.upper_x[3] = (1.f - table1.lookUp(xyz.getMaxXb4())) * dimX + parentBox.lower.x;
					node.upper_y[3] = (1.f - table1.lookUp(xyz.getMaxYb4())) * dimY + parentBox.lower.y;
					node.upper_z[3] = (1.f - table3.lookUp(xyz.getMaxZb4())) * dimZ + parentBox.lower.z;

					return node;
				}
			};

		// ref: floats
		template <>
			class Node<flavor::ref, quantization::zero, quantization::zero, 32, 32, 32> : public BaseNode {
			public:
				typedef NodeStorage<flavor::ref, 32, 32, 32> nodeStorage_t;
				NodeStorage<flavor::ref, 32, 32, 32> xyz;
				static LookupTable<quantization::zero, 0> table1;
				static LookupTable<quantization::zero, 0> table2;
				static LookupTable<quantization::zero, 0> table3;

				void setAABB(const BBox3fa& parentBox,
							 const BBox3fa& box00,
							 const BBox3fa& box10,
							 const BBox3fa& box01,
							 const BBox3fa& box11) {

					const BBox3fa boxes[4] = {box00, box10, box01, box11};

					for (int i = 0; i < 4; ++i) {
						xyz.setMinX(boxes[i].lower.x, i);
						xyz.setMaxX(boxes[i].upper.x, i);
						xyz.setMinY(boxes[i].lower.y, i);
						xyz.setMaxY(boxes[i].upper.y, i);
						xyz.setMinZ(boxes[i].lower.z, i);
						xyz.setMaxZ(boxes[i].upper.z, i);
					}
				}

				void getAABB(BBox3fa& retBox, const BBox3fa& parentBox, const int loc) const {
					retBox.lower.x = xyz.getMinX(loc);
					retBox.upper.x = xyz.getMaxX(loc);
					retBox.lower.y = xyz.getMinY(loc);
					retBox.upper.y = xyz.getMaxY(loc);
					retBox.lower.z = xyz.getMinZ(loc);
					retBox.upper.z = xyz.getMaxZ(loc);

				}

				BVH4::Node getNode(const BBox3fa& parentBox) const {
					BVH4::Node node;

					for (size_t i = 0; i < 4; ++i) {
						node.lower_x[i] = xyz.getMinX(i);
						node.lower_y[i] = xyz.getMinY(i);
						node.lower_z[i] = xyz.getMinZ(i);
						node.upper_x[i] = xyz.getMaxX(i);
						node.upper_y[i] = xyz.getMaxY(i);
						node.upper_z[i] = xyz.getMaxZ(i);
					}

					return node;
				}
			};

	}
}
