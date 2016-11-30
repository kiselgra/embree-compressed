#pragma once

#include <vector>

#define QUANT_MID_0 0.00f
#define QUANT_MID_1 0.40f
#define QUANT_MID_2 0.48f
#define QUANT_MID_3 0.49f
#define QUANT_MID_4 0.50f
#define QUANT_MID_5 0.51f
#define QUANT_MID_6 0.52f
#define QUANT_MID_7 0.60f


#define QUANT_BORDER_0 0.000
#define QUANT_BORDER_1 0.005
#define QUANT_BORDER_2 0.010
#define QUANT_BORDER_3 0.050
#define QUANT_BORDER_4 0.100
#define QUANT_BORDER_5 0.200
#define QUANT_BORDER_6 0.400
#define QUANT_BORDER_7 0.600

namespace embree {
	namespace oriented {

		struct quantization {
			enum { uni, man, man2, zero};
		};

		class BaseLookupTable {
		public:
			inline unsigned char lookUpIdx(const float& val) const {
				unsigned char ret = 0;
				unsigned int end = table.size();

				for (unsigned i = 0; i < table.size(); ++i)
					if (table[i] <= val) ret = i;
					else break;

				return ret;
			}

			inline float lookUp(const unsigned char idx) const {
				if (idx < nValues) return table[idx];
				else std::cerr << "Index out of Range:" << (unsigned int)idx << std::endl;
				return 0.f;
			}

			unsigned nValues;
			std::vector<float> table;
		};


		template <int quant = quantization::uni, int bits = 8>
			class LookupTable : public BaseLookupTable {
			public:
				LookupTable() {
					this->nValues = std::pow(2, bits);
					this->table.resize(nValues);
					this->table.clear();
					for (unsigned i = 0; i < this->nValues; ++i)
						this->table.push_back(static_cast<float>(i) / static_cast<float>(this->nValues));
				}
			};

		template <>
			class LookupTable<quantization::man2, 3> : public BaseLookupTable {
			public:
				LookupTable() {
					this->nValues = 8;
					this->table.resize(8);

					table[0] = QUANT_MID_0;
					table[1] = QUANT_MID_1;
					table[2] = QUANT_MID_2;
					table[3] = QUANT_MID_3;
					table[4] = QUANT_MID_4;
					table[5] = QUANT_MID_5;
					table[6] = QUANT_MID_6;
					table[7] = QUANT_MID_7;
				}
			};

		template <>
			class LookupTable<quantization::man, 3> : public BaseLookupTable {
			public:
				LookupTable() {
					this->nValues = 8;
					this->table.resize(8);

					table[0] = QUANT_BORDER_0;
					table[1] = QUANT_BORDER_1;
					table[2] = QUANT_BORDER_2;
					table[3] = QUANT_BORDER_3;
					table[4] = QUANT_BORDER_4;
					table[5] = QUANT_BORDER_5;
					table[6] = QUANT_BORDER_6;
					table[7] = QUANT_BORDER_7;
				}
			};

		template<int bits>
			class LookupTable<quantization::zero, bits> : public BaseLookupTable {
			public:
				LookupTable() {
					this->nValues = std::pow(2, bits);
					this->table.resize(this->nValues);
					table[0] = 0.f;
					for (size_t i = 1; i < nValues; ++i) 
						table[i] = 1.f;
				}
			};
	}
}
