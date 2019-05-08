#include <gtest/gtest.h>

#include <cstdio>
#include <matrix.h>
#include <matrixutil.h>
#include <clusteringutility.h>
#include <distributions.h>
#include <polynomial.h>
#include <chrono>
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>

#define MATIN "matrixA.in"
#define CATEGORICAL_MATIN "categoricalmatrixA.in"

typedef std::chrono::high_resolution_clock Clock;

TEST(Matrix, TestHouseholderAndQR) {
    Matrix<long double> init({
                              { 1,2,2},
                              {-1,1,2},
                              {-1,0,1},
                              { 1,1,2}
                            });
    Matrix<long double> Q, Qb, R, Rb, A;

    MatrixUtil<long double>::QRDecomposition(init, Q, R);
    A = Q * R;
    ASSERT_EQ(MatrixUtil<long double>::compareEquals(A, init), true);

    MatrixUtil<long double>::HouseholderQR(init, Q, R);

    A = Q * R;
    ASSERT_EQ(MatrixUtil<long double>::compareEquals(A, init), true);
}

TEST(Matrix, CompareAlternatingMinimisationAlgorithms) {
    Matrix<long double> A({
      {1, 2, 3, 2},
      {4, 5, 6, 3},
      {1, 2, 8, 1},
      {4, 4, 7, 4},
      {1, 1, 5, 1},
      {1, 2, 3, 4},	  
    });

    Matrix<long double> X1, Y1, W, ATA = A.transpose()*A;
    MatrixUtil<long double>::AlternatingMinimisationIQRPCA(ATA, 2, W, Y1);
    X1 = A*W.transpose();

    Matrix<long double> X2, Y2;
    MatrixUtil<long double>::AlternatingMinimisationQRPCA(A, 2, X2, Y2);
    Matrix<long double> approx1 = X1*Y1, approx2 = X2*Y2;
    cout << approx1 << endl;
    cout << approx2 << endl;
    ASSERT_EQ(MatrixUtil<long double>::compareEquals(approx1, approx2, 0.01), true);
}

TEST(Matrix, GabiLibraryTest) {
  cout <<setprecision(25) <<fixed;
  Matrix<long double> A1({
    {1, 2, 3, 2},
    {4, 5, 6, 3},
    {1, 2, 8, 1},
    {4, 4, 7, 4},
    {1, 1, 5, 1},
    {1, 2, 3, 4},	  
  });

  Matrix<long double> X1, Y1, W1, ATA1 = A1.transpose()*A1, approx1;
  MatrixUtil<long double>::AlternatingMinimisationIQRPCA2(ATA1, 2, W1, Y1);
  cout << "W1:\n";
  W1.print(25);
  cout << endl << endl;
  cout << "Y1:\n";
  Y1.print(25);
  cout << endl << endl;

  X1 = A1*W1.transpose();
  approx1 = X1 * Y1;
  cout << "approx1:\n";
  approx1.print(25);
  cout << endl << endl;
}

TEST(Matrix, EigenLibraryTest) {
  cout <<setprecision(25) <<fixed;
  Eigen::MatrixXd A(6,4), ATA, W, Y, X, approx;
  A << 1, 2, 3, 2,
       4, 5, 6, 3,
       1, 2, 8, 1,
       4, 4, 7, 4,
       1, 1, 5, 1,
       1, 2, 3, 4;
  ATA = A.transpose() * A;
  MatrixUtil<double>::FactorisedAlternatingMinimisation(ATA, 2, W, Y);
  cout << "W:\n" << W << endl << endl;
  cout << "Y:\n" << Y << endl << endl;
  
  X = A*W.transpose();
  approx = X * Y;
  cout << "approx:\n" << approx << endl << endl;
}

TEST(Matrix, EigendecompositionTest) {
  cout <<setprecision(25) <<fixed;
  Eigen::MatrixXd A(6,4), ATA, W, Y, X, approx;
  A << 1, 2, 3, 2,
       4, 5, 6, 3,
       1, 2, 8, 1,
       4, 4, 7, 4,
       1, 1, 5, 1,
       1, 2, 3, 4;
  ATA = A.transpose() * A;
  MatrixUtil<double>::FactorisedEigenDecomposition(ATA, 2, W, Y, 1.0);
  cout << "W:\n" << W << endl << endl;
  cout << "Y:\n" << Y << endl << endl;
  
  X = A*W.transpose();
  approx = X * Y;
  cout << "approx:\n" << approx << endl << endl;
  cout << "original:\n" << A << endl << endl;
}

TEST(Matrix, EigenChubbyLibraryTest) {
  cout <<setprecision(25) <<fixed;
  Eigen::MatrixXd A(4,6), AAT, L, Y, X, approx;
  A << 1, 4, 1, 4, 1, 1,
       2, 5, 2, 4, 1, 2,
       3, 6, 8, 7, 5, 3,
       2, 3, 1, 4, 1, 4;
   
  AAT = A*A.transpose();
  MatrixUtil<double>::EigenChubbyAlternatingMinimisationIQRPCA(AAT, 2, X, L);
  cout << "X:\n" << X << endl << endl;
  cout << "L:\n" << L << endl << endl;
  
  Y = L.transpose() * A;
  approx = X * Y;
  cout << "approx:\n" << approx << endl << endl;
}
TEST(Matrix, EigendChubbyecompositionTest) {
  cout <<setprecision(25) <<fixed;

  Eigen::MatrixXd A(4,6), AAT, L, Y, X, approx;
  A << 1, 4, 1, 4, 1, 1,
       2, 5, 2, 4, 1, 2,
       3, 6, 8, 7, 5, 3,
       2, 3, 1, 4, 1, 4;
 
  AAT = A*A.transpose();
  MatrixUtil<double>::QRPCAusingChubbyEigendecomposition(AAT, 2, X, L);
  cout << "L:\n" << L << endl << endl;
  cout << "X:\n" << X << endl << endl;
  
  Y = L.transpose() * A;
  approx = X * Y;
  cout << "approx:\n" << approx << endl << endl;
  cout << "original:\n" << A << endl << endl;
}


/**
TEST(Matrix, GabiStressTest) {
  Eigen::MatrixXd A1(20000000, 17);
  Matrix<double> A(20000000, 17);
  Gaussian<double> gauss(0, 65536);
  for (int i = 0; i < A.numRows(); ++i) {
    for (int j = 0; j < A.numCols(); ++j) {
      A.set(i, j,  (i+1)*(j+1) + gauss.sample());
      A1(i,j) = A.get(i,j);
    }
    if (i % 1000000 == 0) {
      for (int j = 0; j < A.numCols(); ++j)
	cout << A.get(i, j) << " ";
      cout << endl;
    }
  }
  cout << "And the A.transpose()*A matrix is:\n";
  Matrix<double> ATA = A.transpose() * A;
  ATA.print(10);
  auto t1 = Clock::now();

  Matrix<double> X1, Y1, W1;
  MatrixUtil<double>::AlternatingMinimisationIQRPCA2(ATA, 7, W1, Y1);

  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
	    << " nanoseconds\n";

  cout << "Matrix W \n";
  W1.print(10);
  cout << "Matrix Y \n";
  Y1.print(10);
    cout << "And the A.transpose()*A matrix is:\n";
  Eigen::MatrixXd ATA2 = A1.transpose() * A1;
  cout << ATA;
  auto t12 = Clock::now();

  Eigen::MatrixXd X12, Y12, W12;
  MatrixUtil<double>::EigenAlternatingMinimisationIQRPCA(ATA2, 7, W12, Y12);

  auto t22 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t22 - t12).count()
	    << " nanoseconds\n";

  cout << "Matrix W \n";
  cout << W12 << endl << endl;
  cout << "Matrix Y \n";
  cout << Y12 << endl << endl;

}
*/

TEST(Matrix, EigenAlternatingStressTest) {
  Eigen::MatrixXd A(5000, 1000);
  Gaussian<long double> gauss(0, 65536);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j)
      A(i, j) = ((i+1)*(j+1)*1.0+ gauss.sample());
    // if (i % 1000000 == 0) {
      // for (int j = 0; j < A.cols(); ++j)
    //	cout << A(i, j) << " ";
    //  cout << endl;
    //}
  }
  cout << "And the A.transpose()*A matrix is:\n";
  Eigen::MatrixXd ATA = A.transpose() * A;
  //cout << ATA;
  cout << "Starting:\n";
  auto t1 = Clock::now();

  Eigen::MatrixXd X1, Y1, W1;
  MatrixUtil<double>::FactorisedAlternatingMinimisation(ATA, 2, W1, Y1);

  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";

  cout << "Matrix W \n";
  //cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  //cout << Y1 << endl << endl;

  t1 = Clock::now();
  
  X1 = A * W1.transpose();

  t2 = Clock::now();
  
  std::cout << "Time taken to multiply A*W1.transpose(): " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";
}

TEST(Matrix, EigenChubbyAlternatingStressTest) {
  Eigen::MatrixXd A(5000, 1000);
  Gaussian<long double> gauss(0, 65536);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j)
      A(i, j) = ((i+1)*(j+1)*1.0 + gauss.sample());
    // if (i % 1000000 == 0) {
      // for (int j = 0; j < A.cols(); ++j)
    //	cout << A(i, j) << " ";
    //  cout << endl;
    //}
  }
  cout << "And the A*A.transpose() matrix is:\n";
  Eigen::MatrixXd AAT = A.transpose()*A;
  //cout << AAT;
  
  cout << "Starting:\n";
  auto t1 = Clock::now();

  Eigen::MatrixXd X1, Y1, L1;
  MatrixUtil<double>::EigenChubbyAlternatingMinimisationIQRPCA(AAT, 2, X1, L1);

  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";

  cout << "Matrix X \n";
  //cout << X1 << endl << endl;
  cout << "Matrix L \n";
  //cout << L1 << endl << endl;

  t1 = Clock::now();
  
  Y1 = L1.transpose()*A.transpose();

  t2 = Clock::now();
  
  std::cout << "Time taken to multiply L1.transpose() * A: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";
}


TEST(Matrix, EigenDecompStressTest) {
  Eigen::MatrixXd A(5000, 1000);
  Gaussian<long double> gauss(0, 65536);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j)
      A(i, j) = ((i+1)*(j+1)*1.0 + gauss.sample());
    // if (i % 1000000 == 0) {
      // for (int j = 0; j < A.cols(); ++j)
    //	cout << A(i, j) << " ";
    //  cout << endl;
    //}
  }
  cout << "And the A.transpose()*A matrix is:\n";
  Eigen::MatrixXd ATA = A.transpose() * A;
  //cout << ATA;
  cout << "Starting:\n";
  auto t1 = Clock::now();

  Eigen::MatrixXd X1, Y1, W1;
  MatrixUtil<double>::FactorisedEigenDecomposition(ATA, 2, W1, Y1);

  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";

  cout << "Matrix W \n";
  //cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  //cout << Y1 << endl << endl;

  t1 = Clock::now();
  
  X1 = A * W1.transpose();

  t2 = Clock::now();
  
  std::cout << "Time taken to multiply A*W1.transpose(): " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";
}

TEST(Matrix, EigenChubbyDecompStressTest) {
  Eigen::MatrixXd A(5000, 1000);
  Gaussian<long double> gauss(0, 65536);
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j)
      A(i, j) = ((i+1)*(j+1)*1.0+ gauss.sample());
    // if (i % 1000000 == 0) {
      // for (int j = 0; j < A.cols(); ++j)
    //	cout << A(i, j) << " ";
    //  cout << endl;
    //}
  }
  cout << "And the A*A.transpose() matrix is:\n";
  Eigen::MatrixXd AAT = A.transpose()*A;
  //cout << AAT;
  
  cout << "Starting:\n";
  auto t1 = Clock::now();

  Eigen::MatrixXd X1, Y1, L1;
  MatrixUtil<double>::QRPCAusingChubbyEigendecomposition(AAT, 2, X1, L1);

  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";

  cout << "Matrix X \n";
  //cout << X1 << endl << endl;
  cout << "Matrix L \n";
  //cout << L1 << endl << endl;

  t1 = Clock::now();
  
  Y1 = L1.transpose()*A.transpose();

  t2 = Clock::now();
  
  std::cout << "Time taken to multiply L1.transpose() * A: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";
}


/**
TEST(Matrix, HousingDatasetGabiMatrix) {
  freopen(CATEGORICAL_MATIN, "r", stdin);

  int N, M;
  scanf("%d", &N); M = N;
  Matrix<long double> ATA(N, M);
  int i, j;
  double val = 0;
  //for (int i = 0; i < N; ++i) {
  //for (int j = 0; j < M; ++j) {
  while (scanf("%d%d%lf", &i, &j, &val) == 3)
    ATA.set(i, j, val);
    //  double val;
    //scanf("%lf", &val);
  // }
  //}

  Matrix<long double> X1, Y1, W1;
  auto t1 = Clock::now();
  MatrixUtil<long double>::AlternatingMinimisationIQRPCA2(ATA, 7, W1, Y1);
  auto t2 = Clock::now();
  std::cout << "Time taken: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";


  cout << "Matrix W \n";
  cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  cout << Y1 << endl << endl;  
  fclose(stdin);
}
*/

TEST(Matrix, MultithreadingTest) {
  cout << "Number of threads: " << Eigen::nbThreads() << endl;
  int N = 500;
  int M = 800;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(M, N);

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd Cinv = Eigen::MatrixXd::Zero(N, N);
  
  Eigen::MatrixXd Ik = Eigen::MatrixXd::Zero(N, N);
  for (int i = 0; i < N; ++i)
    Ik(i,i) = 1;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      A(i,j) = 11*(i+1)*(j+1);
  cout << "Computed A" << endl;
  S = A.transpose() * A;
  cout << "Computed covariance matrix" << endl;
  S = S + Ik; // make it invertible
  cout << "Computed invertible matrix" << endl;
  B = S + Ik*5;
  cout << "Computed second invertible matrix" << endl;
  auto t1 = Clock::now();
  C = S * B;  /// 3 times faster when using 4 threads
  auto t2 = Clock::now();
  
  std::cout << "Time taken to multiply: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6
	    << " miliseconds\n";

  Eigen::LLT<Eigen::MatrixXd> choleskySolver; 
  Eigen::LDLT<Eigen::MatrixXd> choleskySolver2; 
  Eigen::PartialPivLU<Eigen::MatrixXd> LUSolver;
  auto t3 = Clock::now(); 
  LUSolver.compute(C);
  Cinv = LUSolver.solve(Ik);

  //Cinv = C.inverse();

  //choleskySolver.compute(C); /// Sadly it is not optimised by concurrency 
  //Cinv = choleskySolver.solve(Ik);

  //choleskySolver2.compute(C); /// Sadly it is not optimised by concurrency 
  //Cinv = choleskySolver.solve(Ik);

  auto t4 = Clock::now();
  
  std::cout << "Time taken to invert: " 
	    << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e6
	    << " miliseconds\n";
  cout << "Checking that the inverse is corect" << endl;
  assert(Cinv*C == Ik);
  cout << "Done all ok" << endl;
}

TEST(Matrix, HousingDatasetDense) { 
  freopen(CATEGORICAL_MATIN, "r", stdin);

  int N, M;
  //scanf("%d%d", &N, &M);
  scanf("%d", &N); M = N;
  Eigen::MatrixXd ATA = Eigen::MatrixXd::Zero(N,M);
  int i, j;
  double val = 0;
  //for (int i = 0; i < N; ++i) {
  //for (int j = 0; j < M; ++j) {
  while (scanf("%d%d%lf", &i, &j, &val) == 3)
      ATA(i,j) = val;
    //  double val;
    //scanf("%lf", &val);
  // }
  //}
  //cout << ATA << endl;
  Eigen::MatrixXd Y1, W1;

  auto t1 = Clock::now();
  MatrixUtil<double>::FactorisedAlternatingMinimisation(ATA, 7, W1, Y1);
  auto t2 = Clock::now();
  cout << setprecision(30) << fixed;
  cout << "Matrix W \n";
  cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  cout << Y1 << endl << endl;  
  fclose(stdin);
}

/**
TEST(Matrix, HousingDatasetSparse) {
  freopen(CATEGORICAL_MATIN, "r", stdin);

  int N, M;
  scanf("%d", &N);
  M = N;
  std::vector<Eigen::Triplet<double> > listOfEntries;
  //for (int i = 0; i < N; ++i) {
  //  for (int j = 0; j < M; ++j) {
  double val;
  int i, j;
  while (scanf("%d%d%lf", &i, &j, &val) == 3)
    if ( abs(val) > 1e-6 )
      listOfEntries.emplace_back(i, j, val);
      
      // }
  // }
  Eigen::SparseMatrix<double> ATA(N, M);
  ATA.setFromTriplets(listOfEntries.begin(), listOfEntries.end());
  
  Eigen::SparseMatrix<double> Y1, W1;

  auto t1 = Clock::now();
  MatrixUtil<double>::EigenSparseAlternatingMinimisationIQRPCA(ATA, 7, W1, Y1);
  auto t2 = Clock::now();

  cout << "Matrix W \n";
  cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  cout << Y1 << endl << endl;  
}

*/

TEST(Matrix, EigendecompositionDataset) {
 freopen(CATEGORICAL_MATIN, "r", stdin);

  int N, M;
  scanf("%d", &N); M = N;
  Eigen::MatrixXd ATA = Eigen::MatrixXd::Zero(N,M);
  int i, j;
  double val = 0;
  while (scanf("%d%d%lf", &i, &j, &val) == 3)
      ATA(i,j) = val;
  Eigen::MatrixXd Y1, W1;

  MatrixUtil<double>::FactorisedEigenDecomposition(ATA, 7, W1, Y1);

  cout << setprecision(30) << fixed;
  cout << "Matrix W \n";
  cout << W1 << endl << endl;
  cout << "Matrix Y \n";
  cout << Y1 << endl << endl;  
  fclose(stdin);
}
