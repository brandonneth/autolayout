//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"

template <typename T>
T *allocate(std::size_t size)
{
  T *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(
      cudaMallocManaged((void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal));
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(ptr));
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}

using namespace RAJA;
template <std::size_t N>
std::array<idx_t,N> combine_perms(std::array<idx_t, N> perm1, std::array<idx_t, N> perm2) {
  std::array<idx_t,N> perm;

  for(long unsigned int i = 0; i < N; i++) {
    perm[i] = perm2[perm1[i]];
  }

  return perm;
}




int run_2mm() {


  //matrix dimensions
  int ni = 1000;
  int nj = 1000;
  int nk = 1000;
  int nl = 1000;


  //creating Views
  std::cout << "Creating Views\n";
  using VIEW = View<double, Layout<2, int, 1>>;
  VIEW A(new double[ni*nk], Layout<2>(ni, nk));
  VIEW B(new double[nk*nj], Layout<2>(nk, nj));
  VIEW AB(new double[ni*nj], Layout<2>(ni,nj));
  VIEW C(new double[nj*nl], Layout<2>(nj, nl));
  VIEW D(new double[ni*nl], Layout<2>(ni, nl));
  
  //policy for initializing loops 
  using KPol_Init = KernelPolicy<
    statement::For<0, loop_exec,
      statement::For<1, loop_exec,
        statement::Lambda<0>
      >
    >
  >;


  //segments
  auto segi = RangeSegment(0,ni);
  auto segj = RangeSegment(0,nj);
  auto segk = RangeSegment(0,nk);
  auto segl = RangeSegment(0,nl);


  //tuples for the initialization
  auto tupleA = make_tuple(RangeSegment(0,ni), RangeSegment(0,nk));
  auto tupleB = make_tuple(RangeSegment(0,nk), RangeSegment(0,nj));
  auto tupleAB = make_tuple(RangeSegment(0,ni), RangeSegment(0,nj));
  auto tupleC = make_tuple(RangeSegment(0,nj), RangeSegment(0,nl));
  auto tupleD = make_tuple(RangeSegment(0,ni), RangeSegment(0,nl));

 

  auto initA = make_kernel<KPol_Init>(tupleA, [=](auto i, auto j) { A(i,j) = 0.1 * (i + 1.1) / (j + 1.12345);});
  auto initB = make_kernel<KPol_Init>(tupleB, [=](auto i, auto j) { B(i,j) = 0.2 * (i + 1.1) / (j + 1.12345);});
  auto initAB = make_kernel<KPol_Init>(tupleAB, [=](auto i, auto j) { AB(i,j) = 0;});
  auto initC = make_kernel<KPol_Init>(tupleC, [=](auto i, auto j) { C(i,j) = 0.2 * (i + 1.1) / (j + 1.12345);});
  auto initD = make_kernel<KPol_Init>(tupleD, [=](auto i, auto j) { D(i,j) = 0;});
 
  
  //policy for the computation
  using KPol_2MM =
        KernelPolicy<
          statement::For<0, loop_exec,
            statement::For<1, loop_exec,
              statement::For<2, loop_exec,
                statement::Lambda<0>
              >
            >
          >
        >; 

  //tuples for computation
  auto tuple_mm1 = make_tuple(segi, segj, segk);
  auto tuple_mm2 = make_tuple(segi, segl, segj);

  //kernels for the computation
  auto mm1 = make_kernel<KPol_2MM>(tuple_mm1, [=](auto i, auto j, auto k) {AB(i,j) += A(i,k) * B(k,j);});
  auto mm2 = make_kernel<KPol_2MM>(tuple_mm2, [=](auto i, auto j, auto k) {D(i,j) += AB(i,k) * C(k,j);});


  std::cout << "All kernels successfully created.\nSymbolically executing initialization kernels.\n";

  auto initA_accesses = initA.execute_symbolically();
  auto initB_accesses = initB.execute_symbolically();
  auto initAB_accesses = initAB.execute_symbolically();
  auto initC_accesses = initC.execute_symbolically();
  auto initD_accesses = initD.execute_symbolically();


  std::cout << "initing A accesses:\n";
  print_access_list(std::cout, initA_accesses, 1);
  std::cout << "initing B accesses:\n";
  print_access_list(std::cout, initB_accesses, 1);
  std::cout << "initing AB accesses:\n";
  print_access_list(std::cout, initAB_accesses, 1);
  std::cout << "initing C accesses:\n";
  print_access_list(std::cout, initC_accesses, 1);
  std::cout << "initing D accesses:\n";
  print_access_list(std::cout, initD_accesses, 1);

  std::cout << "Symbolically executing computation kernels.\n";
  auto mm1_accesses = mm1.execute_symbolically();
  auto mm2_accesses = mm2.execute_symbolically();

  std::cout << "first multiply accesses:\n";
  print_access_list(std::cout, mm1_accesses, 1); 
  std::cout << "second multiply accesses:\n";
  print_access_list(std::cout, mm2_accesses, 1); 

  std::cout << "\n\nDoes the access information change when we change the layout?\n";

  std::array<idx_t, 2> permA {{1,0}};
  Layout<2> layout_permA = make_permuted_layout( {{ni,nk}}, permA);
  
  VIEW Aperm(new double[ni*nk],  layout_permA);

  auto initAperm = make_kernel<KPol_Init>(tupleA, [=](auto i, auto j) { Aperm(i,j) = 0.1 * (i + 1.1) / (j + 1.12345);});

  auto initAperm_accesses = initAperm.execute_symbolically();
  
  std::cout << "initing A accesses\n";
  print_access_list(std::cout, initA_accesses, 1);
  std::cout << "initing permuted A accesses\n";
  
  print_access_list(std::cout, initAperm_accesses, 1);

  using KPol_2MM_01 =
        KernelPolicy<
          statement::For<1, loop_exec,
            statement::For<0, loop_exec,
              statement::For<2, loop_exec,
                statement::Lambda<0>
              >
            >
          >
        >; 

  
  auto mm1_01 = make_kernel<KPol_2MM_01>(tuple_mm1, [=](auto i, auto j, auto k) {AB(i,j) += A(i,k) * B(k,j);});

  auto mm1_01_accesses = mm1_01.execute_symbolically();

  std::cout << "first multiply accesses:\n";
  print_access_list(std::cout, mm1_accesses, 1); 
  std::cout << "first multiply accesses, outer two loops interchanged\n";
  print_access_list(std::cout, mm1_01_accesses, 1);


  auto layoutPermA = layout_to_perm(A.get_layout());
  auto layoutPermAperm = layout_to_perm(Aperm.get_layout());
  std::cout << "A layout perm:\n";
  std::cout << " " << layoutPermA[0] << " " << layoutPermA[1] << "\n"; 
 std::cout << "Aperm layout perm:\n";
  std::cout << " " << layoutPermAperm[0] << " " << layoutPermAperm[1] << "\n"; 


  std::cout << "initA permuted accesses:\n";
  print_permuted_access_list(std::cout, initA_accesses, 1);
  std::cout << "initAperm permuted accesses:\n";
  print_permuted_access_list(std::cout, initAperm_accesses, 1);

  return 0;
} //run_2mm


int test_combine_perms() {

  std::array<idx_t,3> perm1 {{0,2,1}}; 
  std::array<idx_t,3> perm2 {{2,1,0}};

  std::array<idx_t,3> combined = combine_perms(perm1, perm2);

  std::cout << "Combining 0,2,1 with 2,1,0\n";
  for(const auto& s : combined) {
    std::cout << s << " ";
  } 

 return 0;
}

int test_policy_order() {

  auto order = policy_to_argument_order(statement::Lambda<0>{});

  std::cout << "Order for Lambda<0>, length: " << order.size() << "\n";

  auto order2 = policy_to_argument_order(statement::For<1,loop_exec,statement::Lambda<0>>{});

  std::cout << "Order for for lambda: \n";
  std::cout << order2[0] << "\n";

  using K = KernelPolicy<
    statement::For<0, loop_exec,
      statement::For<1, loop_exec,
        statement::Lambda<0>
      >
    >
  >;

  auto order3 = policy_to_argument_order(K{});
  std::cout << " Order for 2d loop without interchange\n";
  std::cout << order3[0] << " " << order3[1] << "\n";


  using K2 = KernelPolicy<
    statement::For<1, loop_exec,
      statement::For<0, loop_exec,
        statement::Lambda<0>
      >
    >
  >;
  auto order4 = policy_to_argument_order(K2{});
  std::cout << " Order for 2d loop with interchange\n";
  std::cout << order4[0] << " " << order4[1] << "\n";

  return 0;

}
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  run_2mm();

  //test_combine_perms();
 // test_policy_order();
}
