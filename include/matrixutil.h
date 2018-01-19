#ifndef MATRIXUTIL_H
#define MATRIXUTIL_H

#include <matrix.h>
#include <fstream>

template <typename T>
class MatrixUtil
{
  private:
    static const constexpr long double EPS = 1e-20;
  public:
    static bool compareEquals(Matrix<T> A, Matrix<T> B, T epsilon = EPS);
    /// Compare equality between two matrices. Returns true if the component
    /// wise difference never exceeds EPS in absolute value.

    static void QRDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    /// Decomposes the matrix A into a orthonormal matrix Q, whose columns
    /// are orthonormal vectors spanning R^(Q.numColumns). R = Q^T * A.

    static void QRLosslessDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    /// Decomposes the matrix A into a orthonormal matrix Q, whose columns
    /// are orthonormal vectors spanning R^(Q.numColumns). R = Q^T * A.

    static void GaussJordan(Matrix<T> A, Matrix<T> &AI);
    /// Perform the Gauss Jordan elimination on 'A' which produces the inverse
    /// of 'A' in 'AI'. The algorithm produces the identity matrix from A and
    /// equivalently apply the same operations to AI which initially is the
    /// identity matrix.

    static void LeastSquaresQR(Matrix<T> A, Matrix<T> b, Matrix<T> &x, bool lossless = false);
    /// Compute the least squares solution x to the system A*x = b by using
    /// QR decomposition. (A^T*A)x = A^T * b <=> (QR)^T(QR) * x = (QR)^T * b <=>
    /// R^T (Q^TQ) R * x = R^T * Q^Tb <=> R * x = Q^T * b <=> x = R^-1 Q^T * b
    /// the boolean lossless decides whether to compute the solution x to the
    /// system A*x = b using a stable QR decomposition.

    static void HouseholderQR(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R);
    /// Decompose the matrix A into a orthonormal matrix Q and R = Q^T * A.

    static void Reader(Matrix<T> &M, string fileName);
};

template <typename T>
bool MatrixUtil<T>::compareEquals(Matrix<T> A, Matrix<T> B, T eps)
{
    if (A.numCols() != B.numCols() || A.numRows() != B.numRows())
        return false;
    for (int i = 0; i < A.numRows(); ++i)
        for (int j = 0; j < A.numCols(); ++j)
            if (abs(A.get(i, j) - B.get(i, j) > eps)) /// major difference spotted!
            {
                cout << setprecision(16) << fixed;
                cout << A.get(i, j) << " vs \n" << B.get(i, j) << "\n";
                return false;
            }
    return true;
}

template <typename T>
void MatrixUtil<T>::QRDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{   /// Decompose matrix A in the output parameters Q and R such that A = QR.
    Q = Matrix<T>(A.numRows(), A.numCols());
    for (int i = 0; i < A.numCols(); ++i) {
        Matrix<T> qi = A(i); /// extract i'th column
        for (int j = 0; j < i; ++j)
            qi = qi - Q(j).transpose() * A(i) * Q(j);
        qi.normalize();
        Q.setColumn(i, qi);
    }
    R = Q.transpose() * A;
}

template <typename T>
void MatrixUtil<T>::QRLosslessDecomposition(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{
    Q = A;
    for (int i = 0; i < A.numCols(); ++i) {
        Matrix<T> qi = Q(i);
        qi.normalize();
        Q.setColumn(i, qi);
        for (int j = i + 1; j < A.numCols(); ++j)
            Q.setColumn(j, Q(j) - Q(i).transpose() * Q(j) * Q(i));
    }
    R = Q.transpose() * A;
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
                if(A.get(j, i) == 0)
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
void MatrixUtil<T>::HouseholderQR(Matrix<T> A, Matrix<T> &Q, Matrix<T> &R)
{
    int n = A.numCols() - 1;
    int m = A.numRows() - 1;

    Matrix<T> savedA = A;

    auto sign = [](int x) {
        if (x >= 0) return 1;
        return -1;
    };

    for (int k = 0; k < n; ++k) {
        Matrix<T> x = A(k, m, k, k);
        Matrix<T> vk = x;
        vk.set(1,0, vk.get(1,0) + sign(x.get(1, 0)) * x.frobeniusNorm());
        vk.normalize();
        Matrix<T> aux = A(k, m, k, n);
        aux = vk*(vk.transpose()*aux) * (-2.0L);
        for (int i = 0; i < aux.numRows(); ++i)
            for (int j = 0; j < aux.numCols(); ++j)
                A.set(k + i, k + j, A.get(k + i, k + j) + aux.get(i, j));
    }
    R = A;
    GaussJordan(R, Q);
    Q = savedA * Q;
}

template <typename T>
void MatrixUtil<T>::Reader(Matrix<T> &M, string fileName)
{
    ifstream inputStream(fileName.c_str());
    int R, C;
    inputStream >> R >> C;
    M.reset(Matrix<T>(R, C));
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j) {
            T scalar;
            inputStream >> scalar;
            M.set(i, j, scalar);
        }
}

#endif // MATRIXUTIL_H
