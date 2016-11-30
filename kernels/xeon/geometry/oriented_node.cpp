#include "oriented_node.h"

namespace embree {
	namespace oriented {


		// uncompressed nodes, only quantization
		typedef Node<flavor::non,quantization::uni,quantization::uni,3,3,2> uni332non;
		typedef Node<flavor::non,quantization::man,quantization::man2,3,3,2> man332non;

		// uni332non
		template<> LookupTable<quantization::uni, 3> uni332non::table1 = LookupTable<quantization::uni, 3>();
		template<> LookupTable<quantization::uni, 3> uni332non::table2 = LookupTable<quantization::uni, 3>();
		template<> LookupTable<quantization::uni, 2> uni332non::table3 = LookupTable<quantization::uni, 2>();

		// man332non
		template<> LookupTable<quantization::man, 3> man332non::table1 = LookupTable<quantization::man, 3>();
		template<> LookupTable<quantization::man2, 3> man332non::table2 = LookupTable<quantization::man2, 3>();
		template<> LookupTable<quantization::uni, 2> man332non::table3 = LookupTable<quantization::uni, 2>();




		// compressed nodes
		typedef Node<flavor::com,quantization::uni,quantization::uni,3,3,2> uni332a;
		typedef Node<flavor::com,quantization::man,quantization::man2,3,3,2> man332a;

		// uni332a
		template<> LookupTable<quantization::uni, 3> uni332a::table1 = LookupTable<quantization::uni, 3>();
		template<> LookupTable<quantization::uni, 3> uni332a::table2 = LookupTable<quantization::uni, 3>();
		template<> LookupTable<quantization::uni, 2> uni332a::table3 = LookupTable<quantization::uni, 2>();

		// man332a
		template<> LookupTable<quantization::man, 3> man332a::table1 = LookupTable<quantization::man, 3>();
		template<> LookupTable<quantization::man2, 3> man332a::table2 = LookupTable<quantization::man2, 3>();
		template<> LookupTable<quantization::uni, 2> man332a::table3 = LookupTable<quantization::uni, 2>();




		// half slab ompressed nodes
		typedef Node<flavor::mid,quantization::zero,quantization::uni,3,3,2> uni332b;
		typedef Node<flavor::mid,quantization::zero,quantization::man2,3,3,2> man332b;

		// uni332b
		template<> LookupTable<quantization::zero, 3> uni332b::table1 = LookupTable<quantization::zero, 3>();
		template<> LookupTable<quantization::uni, 3> uni332b::table2 = LookupTable<quantization::uni, 3>();
		template<> LookupTable<quantization::uni, 2> uni332b::table3 = LookupTable<quantization::uni, 2>();

		//pre 332b
		template<> LookupTable<quantization::zero, 3> man332b::table1 = LookupTable<quantization::zero, 3>();
		template<> LookupTable<quantization::man2, 3> man332b::table2 = LookupTable<quantization::man2, 3>();
		template<> LookupTable<quantization::uni, 2> man332b::table3 = LookupTable<quantization::uni, 2>();





		// floating point reference node
		typedef Node<flavor::ref,quantization::zero,quantization::zero,32,32,32> refXXX;
		LookupTable<quantization::zero, 0> refXXX::table1 = LookupTable<quantization::zero, 0>();
		LookupTable<quantization::zero, 0> refXXX::table2 = LookupTable<quantization::zero, 0>();
		LookupTable<quantization::zero, 0> refXXX::table3 = LookupTable<quantization::zero, 0>();

	}
}
