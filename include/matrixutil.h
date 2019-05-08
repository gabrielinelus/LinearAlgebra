#ifndef MATRIXUTIL_H
#define MATRIXUTIL_H

#include <matrix.h>
#include <fstream>

#include <eigen/Eigen/Cholesky>
#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>
#include <eigen/Eigen/SparseCholesky>
#include <eigen/Eigen/Eigenvalues>

#include <vector>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


template <typename T>
class MatrixUtil
{
  private:
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixT;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorT; 
    static constexpr long double EPS = 1e-16;
    static constexpr long double tau = 1e-5;   
  public:

    static void QRPCAusingChubbyEigendecomposition(Eigen::MatrixXd  AAT,
						   int              rankK,
						   Eigen::MatrixXd &X,
						   Eigen::MatrixXd &L);
    /**
     * Given the matrix A*A.transpose() (i.e. 'AAT'), and a 'rankK', perform  
     * Quadratically Regularised PCA on matrix A. This will produce the matrix 'L' 
     * such that Y = L.transpose()*A and matrix 'X' such that X*Y is the best rank-k 
     * approximation of A. The method will use eigenvalue-eigenvector.
     * Complexity: O(M^3) actually about 9*M^3 steps.
     */

    static void EigenChubbyAlternatingMinimisationIQRPCA(Eigen::MatrixXd  AAT,
							 int              rankK,
							 Eigen::MatrixXd &X,
							 Eigen::MatrixXd &L,
							 int              iterations = -1);
    /**
     * Given the eigen matrix A*A.transpose() (i.e. 'AAT'), and a 'rankK', perform Inlined  
     * Quadratically Regularised PCA on chubby matrix A. This will produce the eigen matrix 'L' 
     * such that Y = L.transpose()*A and the eigen matrix 'X' such that X*Y is the best rank-k 
     * approximation of A. If sepcified, the algorithm will perform 'iterations'
     * otherwise it will iterate until the iteration improvement is below a loosen 'tau':
     * if ((Lold - Lt).squaredNorm() < tau &&
     *	   (Xold - Xt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: O(M^2*K)
     * Speed: Very fast
     */

    static void FactorisedEigenDecomposition(MatrixT  ATA,
					     int      rankK,
					     MatrixT &W,
					     MatrixT &Y,
					     double   accuracy = 0);
  /**
   * Given the matrix $A^TA$ (i.e. 'ATA'), and a 'rankK', perform  
   * Quadratically Regularised PCA (L2-PCA) on matrix A. This will produce 
   * the matrix 'W' such that $X = AW^T$ and matrix 'Y' such that $XY$ is the
   * best rank-k approximation of A. The method will use eigen-decomposition.
   * If 'accuracy' is provided, then the algorithm will use the minimum rank
   * $k \geq rankK$ which achieves the requested 'accuracy'.  For example
   * $rankK = rank(A^TA)$ if 100$\%$ accuracy is required.
   * Complexity: $\Theta(N^3)$ actually about $9N^3$ steps.
   * Numerical Stability: Good (evaluated by Eigen)
   */

    static void EigenSparseAlternatingMinimisationIQRPCA(Eigen::SparseMatrix<T>  ATA,
							 int                     rankK,
							 Eigen::SparseMatrix<T> &W,
							 Eigen::SparseMatrix<T> &Y,
							 int                     iterations = -1);
    /**
     * Given the eigen sparse matrix A.transpose()*A (i.e. 'ATA'), and a 'rankK', perform Inlined  
     * Quadratically Regularised PCA on matrix A. This will produce the eigen sparse matrix 'W' 
     * such that X = A*W.transpose() and the eigen sparse matrix 'Y' such that X*Y is the best rank-k 
     * approximation of A. If sepcified, the algorithm will perform 'iterations'
     * otherwise it will iterate until the iteration improvement is below a loosen 'tau':
     * if ((Wold - Wt).squaredNorm() < tau &&
     *	   (Yold - Yt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: O(N^2*K)
     * Speed: Average
     * Caution: Do not use unless you have a good reason... Inverse of sparse matrix is most likely dense.
     */

    static void FactorisedAlternatingMinimisation(MatrixT  ATA,
						  int      rankK,
						  MatrixT &W,
						  MatrixT &Y,
						  int      iterations = -1);
    /**
     * Given the matrix $A^TA$ (i.e. 'ATA'), and a 'rankK', perform Inlined  
     * L2-PCA on matrix A. This will produce the eigen matrix 'W' such that 
     * $X = AW^T$ and the eigen matrix 'Y' such that $XY$ is the best rank-k 
     * approximation of A. If sepcified, the algorithm will perform 'iterations'
     * otherwise it will iterate until the iteration improvement is below 'tau':
     * if ((Wold - Wt).squaredNorm() < tau &&
     *	   (Yold - Yt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: $\Theta(N^2rankK)$ where $N = $'ATA.size()'.
     * Speed: Very fast
     */

    static void AlternatingMinimisationIQRPCA2(Matrix<T>  ATA,
					       int        rankK,
					       Matrix<T> &W,
					       Matrix<T> &Y,
					       int        iterations = -1);
    /**
     * Given the matrix A.transpose()*A (i.e. 'ATA'), and a 'rankK', perform Inlined  
     * Quadratically Regularised PCA on matrix A. This will produce the matrix 'W' 
     * such that X = A*W.transpose() and matrix 'Y' such that X*Y is the best rank-k 
     * approximation of A. If sepcified, the algorithm will perform 'iterations'
     * otherwise it will iterate until the iteration improvement is below a loosen 'tau':
     * if ((Wold - Wt).squaredNorm() < tau &&
     *	   (Yold - Yt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: O(N^2*K)
     * Speed: Fast
     */



    static void AlternatingMinimisationIQRPCA(Matrix<T>  ATA,
					      int        rankK,
					      Matrix<T> &W,
					      Matrix<T> &Y,
					      int        iterations = -1);
    /**
     * Given the matrix A.transpose()*A (i.e. 'ATA'), and a 'rankK', perform Inlined  
     * Quadratically Regularised PCA on matrix A. This will produce the matrix 'W' 
     * such that X = A*W.transpose() and matrix 'Y' such that X*Y is the best rank-k 
     * approximation of A. If sepcified, the algorithm will perform 'iterations'
     * otherwise it will iterate until the iteration improvement is below 'tau':
     * if (traceS*(Wold - Wt).squaredNorm() < tau &&
     *            (Yold - Yt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: O(N^2*K)
     * Speed: Average
     */

    static void AlternatingMinimisationQRPCA(Matrix<T>  A,
					     int        rankK,
					     Matrix<T> &X,
					     Matrix<T> &Y,
					     int        iterations = -1);
    /**
     * Given the matrix 'A', and a 'rankK', perform Quadratically Regularised PCA 
     * on matrix A. This will produce the matrix 'X' and matrix 'Y' such that X*Y 
     * is the best rank-k approximation of A. If sepcified, the algorithm will 
     * perform 'iterations' otherwise it will iterate until the iteration improvement
     * is below 'tau':
     * if ((Xold - Xt).squaredNorm() < tau &&
     *	   (Yold - Yt).squaredNorm() < tau)
     *   done = true;
     * Complexity per iteration: O(M*N*K)
     * Speed: Slow!
     */

    static bool compareEquals(Matrix<T> A, Matrix<T> B, T epsilon = EPS);
    /**
     * Compare equality between two matrices. Returns true if the component
     * wise difference never exceeds EPS in absolute value.
     */
    
    static void GaussJordan(Matrix<T> A, Matrix<T> &AI);
    /**
     * Perform the Gauss Jordan elimination on 'A' which produces the inverse
     * of 'A' in 'AI'. The algorithm produces the identity matrix from A and
     * equivalently apply the same operations to AI which initially is the
     * identity matrix.
     */
    
    static void HouseholderQR(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    static void HouseholderQbR(Matrix<T> Ab, Matrix<T> &Qb, Matrix<T> &R);

    static void LeastSquaresQR(Matrix<T> A, Matrix<T> b, Matrix<T> &x, bool lossless = false);
    /**
     * Compute the least squares solution x to the system A*x = b by using
     * QR decomposition. (A^T*A)x = A^T * b <=> (QR)^T(QR) * x = (QR)^T * b <=>
     * R^T (Q^TQ) R * x = R^T * Q^Tb <=> R * x = Q^T * b <=> x = R^-1 Q^T * b
     * the boolean lossless decides whether to compute the solution x to the
     * system A*x = b using a stable QR decomposition.
     */
    
    static void QRDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    /**
     * Decomposes the matrix A into a orthonormal matrix Q, whose columns
     * are orthonormal vectors spanning R^(Q.numColumns). R = Q^T * A.
     */
    
    static void QRLosslessDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    /**
     * Decomposes the matrix A into a orthonormal matrix Q, whose columns
     * are orthonormal vectors spanning R^(Q.numColumns). R = Q^T * A.
     */
    
    static void Reader(Matrix<T> &M, string fileName);
    /**
     * Read a matrix 'M' given a 'fileName' the format of the file should be:
     * on the first line M.numRows() M.numCols() separated by a space and
     * on each of the next M.numRows() lines, M.numCols() type T values.
     */
};

template<typename T>
void MatrixUtil<T>::QRPCAusingChubbyEigendecomposition(Eigen::MatrixXd  AAT,
						   int              rankK,
						   Eigen::MatrixXd &X,
						   Eigen::MatrixXd &Lt)
{
  double initialisationTime = 0;
  double totalAlgorithmTime = 0;
  double multTime = 0;
  double eigenDecompTime = 0;
  double USTime = 0;
  
  auto totalAlgorithmTimeStart = Clock::now();

  // A = USV.transpose = (U*S^1/2) * (S^1/2*V.transpose)
  // A * V * S^(-1/2) = (U * S^1/2)
  //AAT = US^2U.tranpose
  auto initialisationTimeStart = Clock::now();
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(AAT.rows(), AAT.rows());
  Eigen::MatrixXd U = Eigen::MatrixXd::Zero(AAT.rows(), rankK);
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(rankK, rankK);
  Eigen::MatrixXd Diag = Eigen::MatrixXd::Zero(AAT.rows(), AAT.rows());
  Eigen::MatrixXd Sinv = Eigen::MatrixXd::Zero(rankK, rankK);
  Eigen::VectorXd D = Eigen::VectorXd::Zero(AAT.rows());
  auto initialisationTimeEnd = Clock::now();
  initialisationTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(initialisationTimeEnd - initialisationTimeStart).count());

  auto eigenDecompStart = Clock::now();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenDecomposer(AAT);

  L = eigenDecomposer.eigenvectors();
  D = eigenDecomposer.eigenvalues();
  auto eigenDecompEnd = Clock::now();
  eigenDecompTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(eigenDecompEnd - eigenDecompStart).count());
  

  T gamma = 0.0001;


  auto USTimeStart = Clock::now();
  for (int i = 0; i < AAT.rows(); ++i)
    Diag(i,i) = D(i);
  for (int i = 0; i < rankK; ++i) {
    S(i,i) = sqrt(max((T)0, sqrt(D(AAT.rows() - i - 1)) - gamma));
    Sinv(i,i) = 1.0 / S(i,i);
    U.col(i) = L.col(AAT.rows() - i - 1);
  }
  auto USTimeEnd = Clock::now();
  USTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(USTimeEnd - USTimeStart).count());

  auto multTimeStart = Clock::now();
  
  X = U*S;
  Lt = U*Sinv;

  auto multTimeEnd = Clock::now();
  multTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(multTimeEnd - multTimeStart).count());

  auto totalAlgorithmTimeEnd = Clock::now();
  totalAlgorithmTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(totalAlgorithmTimeEnd - totalAlgorithmTimeStart).count());
  bool benchmarking = true;
  if (benchmarking) {
    printf("Benchmarking\n");
    printf("------------\n");
    printf("totalAlgorithmTime:     %.6lf ms\n", totalAlgorithmTime / 1e6);
    printf("initialisationTime:     %.6lf ms\n", initialisationTime / 1e6);
    printf("eigendecompositionTime: %.6lf ms\n", eigenDecompTime / 1e6);
    printf("computing V and S Time: %.6lf ms\n", USTime / 1e6);
    printf("multiplying time:       %.6lf ms\n", multTime / 1e6);
  }

}
    

template<typename T>
void MatrixUtil<T>::EigenChubbyAlternatingMinimisationIQRPCA(Eigen::MatrixXd  AAT,
							     int              rankK,
							     Eigen::MatrixXd &X,
							     Eigen::MatrixXd &L,
							     int              iterations)
{ // This one uses absolute error on L for convergence.
  //cout << "AAT:\n" << AAT << endl << endl;
  double initialisationTime = 0;
  double totalAlgorithmTime = 0;
  
  double minIterationTime = (1LL<<60);
  double avgIterationTime = 0;
  double maxIterationTime = -(1LL<<60);
  
  double convergenceTime = 0;
  double multTimeMin = (1LL<<60);
  double multTimeAvg = 0;
  double multTimeMax = -(1LL<<60);
  
  double inverseTimeMin = (1LL<<60);
  double inverseTimeAvg = 0;
  double inverseTimeMax = -(1LL<<60);

  auto totalAlgorithmTimeStart = Clock::now();
  auto initialisationTimeStart = Clock::now();
  
  Eigen::MatrixXd Xt   = Eigen::MatrixXd::Zero(AAT.rows(), rankK);
  Eigen::MatrixXd Xold = Eigen::MatrixXd::Zero(AAT.rows(), rankK);
  Eigen::MatrixXd Lt   = Eigen::MatrixXd::Zero(AAT.rows(), rankK);
  Eigen::MatrixXd Lold = Eigen::MatrixXd::Zero(AAT.rows(), rankK);
  Eigen::MatrixXd G = AAT, Kt,Ktaux, Ct,Ctaux;
  T traceG = G.trace();
  T avgTr = traceG/(1.0*AAT.rows());
  
  for (int i = 0; i < rankK; ++i)
    Xt(i,i) = 1;
  
  Eigen::MatrixXd Ik = Eigen::MatrixXd::Zero(rankK, rankK);
  for (int i = 0; i < rankK; ++i)
    Ik(i,i) = 1;
  Eigen::MatrixXd Ikgamma = Eigen::MatrixXd::Zero(rankK, rankK);
  Eigen::MatrixXd IkgammaPeTr = Eigen::MatrixXd::Zero(rankK, rankK);
  Eigen::MatrixXd GpeTr = G*(1.0/avgTr);
  T gamma = 0.0001;
  Ikgamma = Ik*gamma;
  IkgammaPeTr = Ik*(gamma/avgTr);
  int currentIteration = 0;
  bool done = false;
  auto initialisationTimeEnd = Clock::now();
  initialisationTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(initialisationTimeEnd - initialisationTimeStart).count());

  /**
  cout << "Xt" << endl;
  cout << Xt  << endl;
  cout << "Ikgamma" << endl;
  cout << Ikgamma << endl;
  cout << "IkgammaPeTr" << endl;
  cout << IkgammaPeTr << endl;
  cout << "G:" << endl;
  cout << G << endl;
  cout << "Avgtr: " << avgTr << endl;
  cout << "GpeTr" << endl;
  cout << GpeTr << endl;
  */
  Eigen::LLT<Eigen::MatrixXd> choleskySolver; 
  while (!done) {
    auto iterationBegin = Clock::now();
    ++currentIteration;
    double crtMult = 0;
    double crtInv = 0;
    
    auto m1 = Clock::now();
    Ktaux = Xt.transpose() * Xt + Ikgamma;
    auto m2 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m2-m1).count());

    //cout << "Ktaux" << endl;
    //cout << Ktaux << endl;
    
    auto i1 = Clock::now();
    choleskySolver.compute(Ktaux);//Xt.transpose() * Yt + Ikgamma);//Ktaux);
    Kt = choleskySolver.solve(Ik);
    //Kt = Ktaux.inverse();
    auto i2 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i2-i1).count());

    //cout << "Kt" << endl;
    //cout << Kt << endl;
    
    auto m3 = Clock::now();
    Lold = Lt;
    Lt = Xt*Kt;

    //cout << "Lt" << endl;
    //cout << Lt << endl;
    
    Ctaux = Lt.transpose()*(GpeTr*Lt) + IkgammaPeTr;
    //cout << "Ctaux" << endl;
    //cout << Ctaux << endl;
    
    auto m4 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m4-m3).count());
   
    auto i3 = Clock::now();
    //Ct = Ctaux.inverse();
    choleskySolver.compute(Ctaux);//Wt*SpeTr*Wt.transpose() + IkgammaPeTr);//Btaux);
    Ct = choleskySolver.solve(Ik);
    auto i4 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i4-i3).count());

    // cout << "Ct" << endl;
    //cout << Ct << endl;
    
    auto m5 = Clock::now();
    Xold = Xt;
    Xt = GpeTr*(Lt*Ct);
    
    //cout << "Xt" << endl;
    // cout << Xt << endl;
    
    auto m6 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m6-m5).count());

    inverseTimeMin = min(inverseTimeMin, crtInv);
    inverseTimeAvg += crtInv;
    inverseTimeMax = max(inverseTimeMax, crtInv);

    multTimeMin = min(multTimeMin, crtMult);
    multTimeAvg += crtMult;
    multTimeMax = max(multTimeMax, crtMult);
    
    if (currentIteration % 100 == 0)
      cout << currentIteration << " " << traceG*(Lold - Lt).squaredNorm() << " " << traceG*(Lold - Lt).squaredNorm()/Lold.squaredNorm() << " " <<  (Lold - Lt).squaredNorm()/Lold.squaredNorm() << " " << (Lold - Lt).squaredNorm() << " " << (Xold - Xt).squaredNorm() << "\n";

    if ( (Lold - Lt).squaredNorm() < tau &&
	 (Xold - Xt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
    auto iterationEnd = Clock::now();
    double itTime =  (std::chrono::duration_cast<std::chrono::nanoseconds>(iterationEnd - iterationBegin).count());
    minIterationTime = min(minIterationTime, itTime);
    avgIterationTime += itTime;
    maxIterationTime = max(maxIterationTime, itTime);
  }
  avgIterationTime /= (1.0*currentIteration);
  multTimeAvg /= (1.0*currentIteration);
  inverseTimeAvg /= (1.0*currentIteration);
  
  auto totalAlgorithmTimeEnd = Clock::now();
  totalAlgorithmTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(totalAlgorithmTimeEnd - totalAlgorithmTimeStart).count());
  bool benchmarking = true;
  if (benchmarking) {
    
    convergenceTime = avgIterationTime * currentIteration;
    printf("Benchmarking\n");
    printf("------------\n");
    printf("number of threads: %d\n", Eigen::nbThreads());
    printf("totalAlgorithmTime:%.6lf ms\n", totalAlgorithmTime / 1e6);
    printf("initialisationTime:%.6lf ms\n", initialisationTime / 1e6); 
    printf("convergenceTime:%.6lf ms\n", convergenceTime / 1e6);
    printf("iterations:%d \n", currentIteration);
    printf("minIterationTime:%.6lf ms\n", minIterationTime / 1e6);
    printf("avgIterationTime:%.6lf ms\n", avgIterationTime / 1e6);
    printf("maxIterationTime:%.6lf ms\n", maxIterationTime / 1e6);
    printf("inverseTimeMin:%.6lf ms\n", inverseTimeMin / 1e6);
    printf("inverseTimeAvg:%.6lf ms\n", inverseTimeAvg / 1e6);
    printf("inverseTimeMax:%.6lf ms\n", inverseTimeMax / 1e6);
    printf("multTimeMin:%.6lf ms\n", multTimeMin / 1e6);
    printf("multTimeAvg:%.6lf ms\n", multTimeAvg / 1e6);
    printf("multTimeMax:%.6lf ms\n", multTimeMax / 1e6);
    
  }
  L = Lt;
  X = Xt;
}

template<typename T>
void MatrixUtil<T>::FactorisedEigenDecomposition(MatrixT  ATA,
						 int      rankK,
						 MatrixT &W,
						 MatrixT &Y,
						 double  accuracy)
{

  double initialisationTime = 0;
  double totalAlgorithmTime = 0;
  double multTime = 0;
  double eigenDecompTime = 0;
  double VSTime = 0;
  
  auto totalAlgorithmTimeStart = Clock::now();

  // A = USV.transpose = (U*S^1/2) * (S^1/2*V.transpose)
  // A * V * S^(-1/2) = (U * S^1/2)
  //ATA = VS^2V.tranpose
  auto initialisationTimeStart = Clock::now();
  MatrixT L    = MatrixT::Zero(ATA.rows(), ATA.rows());
  MatrixT Diag = MatrixT::Zero(ATA.rows(), ATA.rows());
  VectorT D    = VectorT::Zero(ATA.rows());

  auto initialisationTimeEnd = Clock::now();
  initialisationTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(initialisationTimeEnd - initialisationTimeStart).count());

  auto eigenDecompStart = Clock::now();
  Eigen::SelfAdjointEigenSolver<MatrixT> eigenDecomposer(ATA);

  L = eigenDecomposer.eigenvectors();
  D = eigenDecomposer.eigenvalues();
  auto eigenDecompEnd = Clock::now();
  eigenDecompTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(eigenDecompEnd - eigenDecompStart).count());
  

  T gamma = 0.0001;

  T sum = 0;
  T current = 0;
  
  auto VSTimeStart = Clock::now();
  for (int i = 0; i < ATA.rows(); ++i) {
    Diag(i,i) = max((T)0, sqrt(D(ATA.rows() - i - 1)) - gamma);
    sum += Diag(i,i)*Diag(i,i);
  }

  for (int i = 0; i < ATA.rows(); ++i) {
    current += Diag(i,i)*Diag(i,i);
    rankK = max(rankK, i + 1);
    if (current + EPS >= accuracy * sum)
      break;
  }
  
  cerr << "Chosen rank: " << rankK << endl;
  MatrixT V    = MatrixT::Zero(ATA.rows(), rankK);
  MatrixT Sh    = MatrixT::Zero(rankK, rankK);
  MatrixT Shinv = MatrixT::Zero(rankK, rankK);
 

  for (int i = 0; i < rankK; ++i) {
    Sh(i,i) = sqrt(Diag(i,i));
    Shinv(i,i) = 1.0 / Sh(i,i);
    V.col(i) = L.col(ATA.rows() - i - 1);
  }
  auto VSTimeEnd = Clock::now();
  VSTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(VSTimeEnd - VSTimeStart).count());

  //cerr << setprecision(20) << fixed;
  // cerr << "Eigenvectors of ATA:\n" <<  L << endl;
  // cerr << "Top " << rankK << " eigenvectors:\n" << V << endl;
  // cerr << "Eigenvalues of ATA:\n" << Diag << endl;
  // cerr << "Top " << rankK << " eigenvalues:\n" << (S*S) << endl;
  // cerr << "ATA:\n" << ATA << endl;
  // cerr << "Approx:\n" << L*Diag*L.transpose() << endl;
  //cerr << "Expecting identity:\n" << V.transpose() * V  << endl;


  auto multTimeStart = Clock::now();
  
  Y = Sh*V.transpose();
  W = Shinv*V.transpose();

  auto multTimeEnd = Clock::now();
  multTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(multTimeEnd - multTimeStart).count());

  auto totalAlgorithmTimeEnd = Clock::now();
  totalAlgorithmTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(totalAlgorithmTimeEnd - totalAlgorithmTimeStart).count());
  bool benchmarking = true;
  if (benchmarking) {
    printf("Benchmarking\n");
    printf("------------\n");
    printf("totalAlgorithmTime:     %.6lf ms\n", totalAlgorithmTime / 1e6);
    printf("initialisationTime:     %.6lf ms\n", initialisationTime / 1e6);
    printf("eigendecompositionTime: %.6lf ms\n", eigenDecompTime / 1e6);
    printf("computing V and S Time: %.6lf ms\n", VSTime / 1e6);
    printf("multiplying time:       %.6lf ms\n", multTime / 1e6);
  }

}

template<typename T>
void MatrixUtil<T>::EigenSparseAlternatingMinimisationIQRPCA(Eigen::SparseMatrix<T>  ATA,
							     int                     rankK,
							     Eigen::SparseMatrix<T> &W,
							     Eigen::SparseMatrix<T> &Y,
							     int                     iterations)
{ // This one uses absolute error on W for convergence.
  //cout << "ATA:\n" << ATA << endl << endl;

  double initialisationTime = 0;
  double totalAlgorithmTime = 0;
  
  double minIterationTime = (1LL<<60);
  double avgIterationTime = 0;
  double maxIterationTime = -(1LL<<60);
  
  double convergenceTime = 0;
  double multTimeMin = (1LL<<60);
  double multTimeAvg = 0;
  double multTimeMax = -(1LL<<60);
  
  double inverseTimeMin = (1LL<<60);
  double inverseTimeAvg = 0;
  double inverseTimeMax = -(1LL<<60);
  
  auto totalAlgorithmTimeStart = Clock::now();
  auto initialisationTimeStart = Clock::now();
  
  Eigen::SparseMatrix<T> Yt(rankK, ATA.rows());
  Eigen::SparseMatrix<T> Yold(rankK, ATA.rows());
  Eigen::SparseMatrix<T> Wt(rankK, ATA.rows());
  Eigen::SparseMatrix<T> Wold(rankK, ATA.rows());
  Eigen::SparseMatrix<T> S = ATA, Zt,Ztaux, Bt,Btaux;
  T traceS = 0;
  for (int i = 0; i < S.outerSize(); ++i)
    traceS += S.coeff(i, i);
  T avgTr = traceS / (1.0*ATA.rows());
  Eigen::SparseMatrix<T> SpeTr = S*(1.0/avgTr);

  // We need an initial guess for the matrix Y
  std::vector<Eigen::Triplet<T> > listOfEntries;
  for (int i = 0; i < rankK; ++i)
    listOfEntries.emplace_back(i, i, i);
  Yt.setFromTriplets(listOfEntries.begin(), listOfEntries.end());
  
  Eigen::SparseMatrix<T> Ik(rankK, rankK);
  Ik.setIdentity();
  Eigen::SparseMatrix<T> Ikgamma(rankK, rankK);
  Eigen::SparseMatrix<T> IkgammaPeTr(rankK, rankK);
  T gamma = 0.0001;
  Ikgamma = Ik*gamma;
  IkgammaPeTr = Ik*(gamma/avgTr);
  int currentIteration = 0;
  bool done = false;
  auto initialisationTimeEnd = Clock::now();
  initialisationTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(initialisationTimeEnd - initialisationTimeStart).count());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<T> > choleskySolver;
  while (!done) {
    auto iterationBegin = Clock::now();
    ++currentIteration;
    double crtMult = 0;
    double crtInv = 0;
    
    auto m1 = Clock::now();
    Ztaux = Yt * Yt.transpose() + Ikgamma;
    auto m2 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m2-m1).count());

    auto i1 = Clock::now();
    choleskySolver.compute(Ztaux);
    Zt = choleskySolver.solve(Ik); // solve Ztaux * Zt = Ik  i.e. Zt = Ztaux^(-1) 
    auto i2 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i2-i1).count());
    
    auto m3 = Clock::now();
    Wold = Wt;
    Wt = Zt*Yt;
    
    Btaux = Wt*SpeTr*Eigen::SparseMatrix<T>(Wt.transpose()) + IkgammaPeTr;

    auto m4 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m4-m3).count());
   
    auto i3 = Clock::now();
    choleskySolver.compute(Btaux);
    Bt = choleskySolver.solve(Ik); // solve Btaux * Bt = Ik  i.e. Bt = Btaux^(-1) 
    auto i4 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i4-i3).count());
    
    auto m5 = Clock::now();
    Yold = Yt;
    Yt = Bt*Wt*SpeTr;
    auto m6 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m6-m5).count());

    inverseTimeMin = min(inverseTimeMin, crtInv);
    inverseTimeAvg += crtInv;
    inverseTimeMax = max(inverseTimeMax, crtInv);

    multTimeMin = min(multTimeMin, crtMult);
    multTimeAvg += crtMult;
    multTimeMax = max(multTimeMax, crtMult);
    
    if (currentIteration % 10000 == 0)
      cout << currentIteration << " " << traceS*(Wold - Wt).squaredNorm() << " " << traceS*(Wold - Wt).squaredNorm()/Wold.squaredNorm() << " " <<  (Wold - Wt).squaredNorm()/Wold.squaredNorm() << " " << (Wold - Wt).squaredNorm() << " " << (Yold - Yt).squaredNorm() << "\n";

    if ( (Wold - Wt).squaredNorm() < tau &&
	 (Yold - Yt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
 
    auto iterationEnd = Clock::now();
    double itTime =  (std::chrono::duration_cast<std::chrono::nanoseconds>(iterationEnd - iterationBegin).count());
    minIterationTime = min(minIterationTime, itTime);
    avgIterationTime += itTime;
    maxIterationTime = max(maxIterationTime, itTime);
  }
  avgIterationTime /= (1.0*currentIteration);
  multTimeAvg /= (1.0*currentIteration);
  inverseTimeAvg /= (1.0*currentIteration);
  
  auto totalAlgorithmTimeEnd = Clock::now();
  totalAlgorithmTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(totalAlgorithmTimeEnd - totalAlgorithmTimeStart).count());
  bool benchmarking = true;
  if (benchmarking) {
    convergenceTime = avgIterationTime * currentIteration;
    printf("Benchmarking\n");
    printf("------------\n");
    printf("totalAlgorithmTime:%.6lf ms\n", totalAlgorithmTime / 1e6);
    printf("initialisationTime:%.6lf ms\n", initialisationTime / 1e6); 
    printf("convergenceTime:%.6lf ms\n", convergenceTime / 1e6);
    printf("iterations:%d \n", currentIteration);
    printf("minIterationTime:%.6lf ms\n", minIterationTime / 1e6);
    printf("avgIterationTime:%.6lf ms\n", avgIterationTime / 1e6);
    printf("maxIterationTime:%.6lf ms\n", maxIterationTime / 1e6);
    printf("inverseTimeMin:%.6lf ms\n", inverseTimeMin / 1e6);
    printf("inverseTimeAvg:%.6lf ms\n", inverseTimeAvg / 1e6);
    printf("inverseTimeMax:%.6lf ms\n", inverseTimeMax / 1e6);
    printf("multTimeMin:%.6lf ms\n", multTimeMin / 1e6);
    printf("multTimeAvg:%.6lf ms\n", multTimeAvg / 1e6);
    printf("multTimeMax:%.6lf ms\n", multTimeMax / 1e6);
    
  }
  W = Wt;
  Y = Yt;
}

template<typename T>
void MatrixUtil<T>::FactorisedAlternatingMinimisation(MatrixT  ATA,
						      int      rankK,
						      MatrixT &W,
						      MatrixT &Y,
						      int      iterations)
{ // This one uses absolute error on W for convergence.
  //cout << "ATA:\n" << ATA << endl << endl;
  double initialisationTime = 0;
  double totalAlgorithmTime = 0;
  
  double minIterationTime = (1LL<<60);
  double avgIterationTime = 0;
  double maxIterationTime = -(1LL<<60);
  
  double convergenceTime = 0;
  double multTimeMin = (1LL<<60);
  double multTimeAvg = 0;
  double multTimeMax = -(1LL<<60);
  
  double inverseTimeMin = (1LL<<60);
  double inverseTimeAvg = 0;
  double inverseTimeMax = -(1LL<<60);

  auto totalAlgorithmTimeStart = Clock::now();
  auto initialisationTimeStart = Clock::now();
  
  MatrixT Yt   = MatrixT::Zero(rankK, ATA.rows());
  MatrixT Yold = MatrixT::Zero(rankK, ATA.rows());
  MatrixT Wt   = MatrixT::Zero(rankK, ATA.rows());
  MatrixT Wold = MatrixT::Zero(rankK, ATA.rows());
  MatrixT S = ATA, Zt,Ztaux, Bt,Btaux;

  T traceS = S.trace();
  T avgTr = traceS/(1.0*ATA.rows());
  for (int i = 0; i < rankK; ++i)
    Yt(i,i) = 1;
  
  MatrixT Ik = MatrixT::Zero(rankK, rankK);
  for (int i = 0; i < rankK; ++i)
    Ik(i,i) = 1;

  MatrixT Ikgamma = MatrixT::Zero(rankK, rankK);
  MatrixT IkgammaPeTr = MatrixT::Zero(rankK, rankK);
  MatrixT SpeTr = S*(1.0/avgTr);

  T gamma = 0.0001;
  Ikgamma = Ik*gamma;
  IkgammaPeTr = Ik*(gamma/avgTr);

  int currentIteration = 0;
  bool done = false;
  auto initialisationTimeEnd = Clock::now();
  initialisationTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(initialisationTimeEnd - initialisationTimeStart).count());

  /**
  cout << "Yt" << endl;
  cout << Yt << endl;
  cout << "Ikgamma" << endl;
  cout << Ikgamma << endl;
  cout << "IkgammaPeTr" << endl;
  cout << IkgammaPeTr << endl;
  cout << "S:" << endl;
  cout << S << endl;
  cout << "Avgtr: " << avgTr << endl;
  cout << "SpeTr" << endl;
  cout << SpeTr << endl;
  */
  
  Eigen::LLT<MatrixT> choleskySolver; 
  while (!done) {
    auto iterationBegin = Clock::now();
    ++currentIteration;
    double crtMult = 0;
    double crtInv = 0;
    
    auto m1 = Clock::now();
    //Ztaux = Yt * Yt.transpose() + Ikgamma;

    //cout << "Ztaux" << endl;
    //cout << Ztaux << endl;

    auto m2 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m2-m1).count());
      
    auto i1 = Clock::now();
    choleskySolver.compute(Yt * Yt.transpose() + Ikgamma);//Ztaux);
    Zt = choleskySolver.solve(Ik);
    //Zt = Ztaux.inverse();
    auto i2 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i2-i1).count());

    //cout << "Zt" << endl;
    //cout << Zt << endl;
    
    auto m3 = Clock::now();
    Wold = Wt;
    Wt = Zt*Yt;

    //cout << "Wt" << endl;
    //cout << Wt << endl;
    
    //Btaux = Wt*SpeTr*Wt.transpose() + IkgammaPeTr;
    //cout << "Btaux" << endl;
    //cout << Btaux << endl;
    
    auto m4 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m4-m3).count());
   
    auto i3 = Clock::now();
    //Bt = Btaux.inverse();
    choleskySolver.compute(Wt*SpeTr*Wt.transpose() + IkgammaPeTr);//Btaux);
    Bt = choleskySolver.solve(Ik);
    auto i4 = Clock::now();
    crtInv += (std::chrono::duration_cast<std::chrono::nanoseconds>(i4-i3).count());

    //cout << "Bt" << endl;
    //cout << Bt << endl;
    
    auto m5 = Clock::now();
    Yold = Yt;
    Yt = Bt*Wt*SpeTr;

    //cout << "Yt" << endl;
    //cout << Yt << endl;
    
    auto m6 = Clock::now();
    crtMult += (std::chrono::duration_cast<std::chrono::nanoseconds>(m6-m5).count());

    inverseTimeMin = min(inverseTimeMin, crtInv);
    inverseTimeAvg += crtInv;
    inverseTimeMax = max(inverseTimeMax, crtInv);

    multTimeMin = min(multTimeMin, crtMult);
    multTimeAvg += crtMult;
    multTimeMax = max(multTimeMax, crtMult);
    
    if (currentIteration % 100 == 0)
      cout << currentIteration << " " << traceS*(Wold - Wt).squaredNorm() << " " << traceS*(Wold - Wt).squaredNorm()/Wold.squaredNorm() << " " <<  (Wold - Wt).squaredNorm()/Wold.squaredNorm() << " " << (Wold - Wt).squaredNorm() << " " << (Yold - Yt).squaredNorm() << "\n";

    if ( (Wold - Wt).squaredNorm() < tau &&
	 (Yold - Yt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
    auto iterationEnd = Clock::now();
    double itTime =  (std::chrono::duration_cast<std::chrono::nanoseconds>(iterationEnd - iterationBegin).count());
    minIterationTime = min(minIterationTime, itTime);
    avgIterationTime += itTime;
    maxIterationTime = max(maxIterationTime, itTime);
  }
  avgIterationTime /= (1.0*currentIteration);
  multTimeAvg /= (1.0*currentIteration);
  inverseTimeAvg /= (1.0*currentIteration);
  
  auto totalAlgorithmTimeEnd = Clock::now();
  totalAlgorithmTime = (std::chrono::duration_cast<std::chrono::nanoseconds>(totalAlgorithmTimeEnd - totalAlgorithmTimeStart).count());
  bool benchmarking = true;
  if (benchmarking) {
    
    convergenceTime = avgIterationTime * currentIteration;
    printf("Benchmarking\n");
    printf("------------\n");
    printf("number of threads: %d\n", Eigen::nbThreads());
    printf("totalAlgorithmTime:%.6lf ms\n", totalAlgorithmTime / 1e6);
    printf("initialisationTime:%.6lf ms\n", initialisationTime / 1e6); 
    printf("convergenceTime:%.6lf ms\n", convergenceTime / 1e6);
    printf("iterations:%d \n", currentIteration);
    printf("minIterationTime:%.6lf ms\n", minIterationTime / 1e6);
    printf("avgIterationTime:%.6lf ms\n", avgIterationTime / 1e6);
    printf("maxIterationTime:%.6lf ms\n", maxIterationTime / 1e6);
    printf("inverseTimeMin:%.6lf ms\n", inverseTimeMin / 1e6);
    printf("inverseTimeAvg:%.6lf ms\n", inverseTimeAvg / 1e6);
    printf("inverseTimeMax:%.6lf ms\n", inverseTimeMax / 1e6);
    printf("multTimeMin:%.6lf ms\n", multTimeMin / 1e6);
    printf("multTimeAvg:%.6lf ms\n", multTimeAvg / 1e6);
    printf("multTimeMax:%.6lf ms\n", multTimeMax / 1e6);
    
  }
  W = Wt;
  Y = Yt;
}

template <typename T>
void MatrixUtil<T>::AlternatingMinimisationIQRPCA2(Matrix<T>  ATA,
						   int        rankK,
						   Matrix<T> &W,
						   Matrix<T> &Y,
						   int        iterations)
{ // This one uses absolute error on W for convergence.
  Matrix<T> Yt(rankK, ATA.numRows()), Yold(rankK, ATA.numRows());
  Matrix<T> Wt(rankK, ATA.numRows()), Wold(rankK, ATA.numRows());
  Matrix<T> S = ATA, Zt,Ztaux, Bt,Btaux;
  T traceS = S.trace();

  T avgTr = traceS/(1.0*ATA.numRows());
  Matrix<T> SpeTr(ATA.numRows(), ATA.numRows());
  SpeTr = S*(1.0/avgTr);

  // We need an initial guess for the matrix Y
  for (int i = 0; i < rankK; ++i)
    Yt.set(i, i, 1);
  
  Matrix<T> Ik(rankK, rankK);
  for (int i = 0; i < rankK; ++i)
    Ik.set(i, i, 1);

  T gamma = 0.0001;
  Matrix<T> Ikgamma(rankK, rankK);
  Matrix<T> IkgammaPeTr(rankK, rankK);
  
  Ikgamma = Ik*gamma;
  IkgammaPeTr = Ik*(gamma/avgTr);
  int currentIteration = 0;
  bool done = false;
  while (!done) {
    ++currentIteration;

    Ztaux = Yt*Yt.transpose() + Ikgamma;
    MatrixUtil<T>::GaussJordan(Ztaux, Zt);
  
    Wold = Wt;
    Wt = Zt*Yt;

    Btaux = Wt*SpeTr*Wt.transpose() + IkgammaPeTr;
    MatrixUtil<T>::GaussJordan(Btaux, Bt);

    Yold = Yt;
    Yt = Bt*Wt*SpeTr;

    if (currentIteration % 10000 == 0)
      cout << currentIteration << " " << traceS*(Wold - Wt).squaredNorm() << " " << traceS*(Wold - Wt).squaredNorm()/Wold.squaredNorm() << " " << (Wold - Wt).squaredNorm() << " " << (Yold - Yt).squaredNorm() << "\n";
 
    if ( (Wold - Wt).squaredNorm() < tau &&
       	 (Yold - Yt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
  }
  W = Wt;
  Y = Yt;
}

template <typename T>
void MatrixUtil<T>::AlternatingMinimisationIQRPCA(Matrix<T>  ATA,
						  int        rankK,
						  Matrix<T> &W,
						  Matrix<T> &Y,
						  int        iterations)
{ // This one uses absolute error on X for convergence.
  Matrix<T> Yt(rankK, ATA.numRows()), Yold(rankK, ATA.numRows());
  Matrix<T> Wt(rankK, ATA.numRows()), Wold(rankK, ATA.numRows());
  Matrix<T> S = ATA, Zt, Bt;
  T traceS = S.trace();

  T avgTr = traceS/(1.0*ATA.numRows());
  Matrix<T> SpeTr(ATA.numRows(), ATA.numRows());
  SpeTr = S*(1.0/avgTr);

  // We need an initial guess for the matrix Y
  for (int i = 0; i < rankK; ++i)
    Yt.set(i, i, 1); 
  
  Matrix<T> Ik(rankK, rankK);
  for (int i = 0; i < rankK; ++i)
    Ik.set(i, i, 1);
  
  T gamma = 0.0001;
  Matrix<T> Ikgamma(rankK, rankK);
  Matrix<T> IkgammaPeTr(rankK, rankK);
  Ikgamma = Ik * gamma;
  IkgammaPeTr = Ik * (gamma/avgTr);
  int currentIteration = 0;
  bool done = false;
  while (!done) {
    ++currentIteration;

    GaussJordan(Yt*Yt.transpose() + Ikgamma, Zt);
    Wold = Wt;
    Wt = Zt*Yt;

    GaussJordan(Wt*SpeTr*Wt.transpose() + IkgammaPeTr, Bt);
    Yold = Yt;
    Yt = Bt*Wt*SpeTr;

    if (currentIteration % 10000 == 0)
      cout << currentIteration << " " << traceS*(Wold - Wt).squaredNorm() << " " << (Wold - Wt).squaredNorm() << " " << (Yold - Yt).squaredNorm() << "\n";
    
    if (traceS*(Wold - Wt).squaredNorm() < tau &&
       	       (Yold - Yt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
  }
  W = Wt;
  Y = Yt;
}

template<typename T>
void MatrixUtil<T>::AlternatingMinimisationQRPCA(Matrix<T>  A,
						 int        rankK,
						 Matrix<T> &X,
						 Matrix<T> &Y,
						 int        iterations)
{
  Matrix<T> Yt(rankK, A.numCols()), Yold(rankK, A.numCols());
  Matrix<T> Xt(A.numRows(), rankK), Xold(A.numRows(), rankK);
  Matrix<T> Zt, Bt;
  
  // We need an initial guess for the matrix Y
  for (int i = 0; i < rankK; ++i)
    Yt.set(i, i, 1); 

  Matrix<T> Ik(rankK, rankK);
  for (int i = 0; i < rankK; ++i)
    Ik.set(i, i, 1);

  T gamma = 0.0001;
  Matrix<T> Ikgamma(rankK, rankK);
  Ikgamma = Ik * gamma;
  int currentIteration = 0;
  bool done = false;
  while (!done) {
    ++currentIteration;

    GaussJordan(Yt*Yt.transpose() + Ikgamma, Zt);
    Xold = Xt;
    Xt = A*Yt.transpose()*Zt;

    GaussJordan(Xt.transpose()*Xt + Ikgamma, Bt);
    Yold = Yt;
    Yt = Bt*Xt.transpose()*A;

    if ((Xold - Xt).squaredNorm() < tau &&
	(Yold - Yt).squaredNorm() < tau)
      done = true;
    if (iterations == currentIteration)
      done = true;
  }
  X = Xt;
  Y = Yt;
  
}

template <typename T>
bool MatrixUtil<T>::compareEquals(Matrix<T> A, Matrix<T> B, T eps)
{
    if (A.numCols() != B.numCols() || A.numRows() != B.numRows())
        return false;
    for (int i = 0; i < A.numRows(); ++i)
        for (int j = 0; j < A.numCols(); ++j)
            if (abs(A.get(i, j) - B.get(i, j)) > eps) /// major difference spotted!
            {
                cout << setprecision(16) << fixed;
                cout << A.get(i, j) << " vs \n" << B.get(i, j) << "\n";
                return false;
            }
    return true;
}

template <typename T>
void MatrixUtil<T>::GaussJordan(Matrix<T> A, Matrix<T> &AI)
{   /// Invert a square matrix A into the output parameter AI
    assert(A.numRows() == A.numCols() && "inverse does not exist");
    AI = Matrix<T>(A.numRows(), A.numCols());
    for (int i = 0; i < AI.numRows(); ++i)
        AI.set(i, i, 1);
    for (int i = 0; i < A.numRows(); ++i) {
        if (A.get(i, i) == 0) {
            /// we need a new row here
            int j = i;
            while (++j < A.numRows())
                if(A.get(j, i) != 0)
                    break;
            assert( j != A.numRows() && "inverse does not exist" );
            A.rowSwap(i, j);
            AI.rowSwap(i, j);
        }
        AI.rowScale(i, 1.0/A.get(i, i));
        A.rowScale(i, 1.0/A.get(i, i));
        for (int j = i + 1; j < A.numRows(); ++j) {
            AI.addToRow(j, AI[i] * (-A.get(j, i)));
            A.addToRow(j, A[i] * (-A.get(j, i)) );
        }
    }
    /// We are now in an upper unitriangular shape
    for (int i = (int)A.numRows() - 1; i >= 0; --i)
        for (int j = i - 1; j >= 0; --j) {
            AI.addToRow(j, AI[i] * (-A.get(j, i)));
            A.addToRow(j, A[i] * (-A.get(j,i)));
        }
    /// We now have A = Identity matrix, AI = inverse matrix
}

template <typename T>
void MatrixUtil<T>::HouseholderQR(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{
    int n = A.numCols() - 1;
    int m = A.numRows() - 1;

    Matrix<T> savedA(A);

    auto sign = [](T x) -> T{
        if (x >= 0) return 1;
        return -1;
    };


    for (int k = 0; k <= n; ++k) {
        Matrix<T> x = A(k, m, k, k);

        Matrix<T> e1(m - k + 1, 1);
        e1.set(0, 0, 1);

        Matrix<T> vk = e1 * sign(x.get(0, 0))*x.frobeniusNorm() + x;
        vk.normalize();
        A.setSubmatrix(k, m, k, n, A(k, m, k, n) - vk*(vk.transpose()*A(k, m, k, n)) * 2.0L);
    }

    R = A(0, n, 0, n);
    Matrix<T> RI;
    GaussJordan(R, RI);
    Q = savedA * RI;
}

template <typename T>
void MatrixUtil<T>::HouseholderQbR(Matrix<T> Ab, Matrix<T> &Qb, Matrix<T> &R)
{
    int n = Ab.numCols() - 1;
    int m = Ab.numRows() - 1;


    auto sign = [](T x) -> T{
        if (x >= 0) return 1;
        return -1;
    };

    for (int k = 0; k <= n; ++k) {
        Matrix<T> x = Ab(k, m, k, k);

        Matrix<T> e1(m - k + 1, 1);
        e1.set(0, 0, 1);

        Matrix<T> vk = e1 * sign(x.get(0, 0))*x.frobeniusNorm() + x;
        vk.normalize();
        Ab.setSubmatrix(k, m, k, n, Ab(k, m, k, n) - vk*(vk.transpose()*Ab(k, m, k, n)) * 2.0L);
    }
    R = Ab(0, n - 1, 0, n - 1);
    Qb = Ab(0, n - 1, n, n);
}

template <typename T>
void MatrixUtil<T>::LeastSquaresQR(Matrix<T> A, Matrix<T> b, Matrix<T> &x, bool lossless)
{   /// A * x = b <=> Q*R * x = b <=> R * x = Q^T*b <=> x = R^-1 * Q^T *b
    Matrix<T> Q, R, RI;
    if (lossless == false)
        QRDecomposition(A, Q, R);
    else
        QRLosslessDecomposition(A, Q, R);
    GaussJordan(R, RI);
    x = RI * Q.transpose() * b;
}

template <typename T>
void MatrixUtil<T>::QRDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{   /// Decompose matrix A in the output parameters Q and R such that A = QR.
    int m = A.numRows(), n = A.numCols();

    Q = Matrix<T>(m, n);
    R = Matrix<T>(n, n);
    Matrix<T> V = A;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            R.set(i, j, (Q(i).transpose() * A(j)).get(0, 0));
            V.setColumn(j, V(j) - Q(i) * R.get(i, j));
        }
        R.set(j, j, V(j).frobeniusNorm());
        Q.setColumn(j, V(j) / R.get(j, j));
    }
}

template <typename T>
void MatrixUtil<T>::QRLosslessDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{
    int m = A.numRows(), n = A.numCols();

    Q = Matrix<T>(m, n);
    R = Matrix<T>(n, n);
    Matrix<T> V = A;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            R.set(i, j, (Q(i).transpose() * V(j)).get(0, 0));
            V.setColumn(j, V(j) - Q(i) * R.get(i, j));
        }
        R.set(j, j, V(j).frobeniusNorm());
        Q.setColumn(j, V(j) / R.get(j, j));
    }
}

template <typename T>
void MatrixUtil<T>::Reader(Matrix<T> &M, string fileName)
{
    ifstream inputStream(fileName.c_str());
    int R, C;
    inputStream >> R >> C;
    M = Matrix<T>(R, C);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) {
            T scalar;
            inputStream >> scalar;
            M.set(i, j, scalar);
        }
}

#endif // MATRIXUTIL_H
