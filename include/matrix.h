#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cmath>

using namespace std;

template <typename T>
class Matrix
{
  private:
    std::vector<vector<T> > d_rows;
    int d_numRows;
    int d_numCols;

  public:
/// Creators
    Matrix();
    ///Default constructor

    Matrix(int rows, int columns);
    /// Constructs an empty matrix with 'rows' rows and 'columns' columns.

    Matrix(const vector<vector<T> >& other);
    /// Constructs a matrix from a C++ vector of vectors.

    Matrix(const Matrix& other);
    /// Copy constructs a matrix with the same number of rows and columns
    /// as the given one.

    Matrix& operator=(const Matrix& other);
    /// Construct the current object with the specified 'other' matrix.

/// Manipulators

    void rowScale(int i, T scalar);
    /// scale row i by T

    void colScale(int j, T scalar);
    /// scale column j by T

    void rowSwap(int i, int j);
    /// swap two rows of a matrix

    void colSwap(int i, int j);
    /// swap two columns of a matrix

    void addToRow(int i, const Matrix<T>& other);
    /// add to a row of a matrix a given row in matrix form

    void addToCol(int j, const Matrix<T>& other);
    /// add to the column of a matrix a given column in matrix form

    void normalize();
    /// Divide by the Frobenius Norm.

    void setColumn(int index, const Matrix<T>& other);
    /// Set the column 'index' to equal 'other'.

    void setRow(int index, const Matrix<T>& other);
    /// Set the row 'index' to equal other.

    void set(int row, int col, T value);
    /// Set the value at specified 'row' and specified 'col' to 'value'.

    Matrix<T> transpose();
    /// Return the transpose of 'this' Matrix. If it's cached then just returns it.

    T frobeniusNorm();
    /// Return the entry square norm

/// Accessors

    const void print(int indent) const;
    /// Print the matrix with the given amount of right indentation.

    const T get(int row, int col) const;
    /// Get the value at specified 'row' and specified 'col'.

    const int numRows() const;
    /// Get the number of rows.

    const int numCols() const;
    /// Get the number of columns.

    Matrix<T> operator[](int index) const;
    /// Get the 'index'th row of the Matrix in a Matrix row form.

    Matrix<T> operator()(int upRow, int downRow, int leftCol, int rightCol);
    /// Get the submatrix defined by rows [upRow, downRow] and columns [leftCol, rightCol].

    Matrix<T> operator()(int index) const;
    /// Get the 'index'th column of the Matrix in a Matrix column form.

/// Free operators
    bool operator == (const Matrix<T>& other);
    /// Returns if the two matrices are equal

    bool operator != (const Matrix<T>& other);
    /// Returns if the two matrices are different

    Matrix<T>& operator *= (const Matrix<T>& rhs);
    /// Multiply 'this' by 'rhs' matrix.

    Matrix<T>& operator *= (const T& rhs);
    /// Multiply 'this' by 'rhs' scalar.

    Matrix<T>& operator += (const Matrix<T>& rhs);
    /// Add to 'this' matrix the 'rhs' matrix

    Matrix<T>& operator -= (const Matrix<T>& rhs);
    /// Subtract from 'this' matrix the 'rhs' matrix

    Matrix<T>& operator /= (const T& rhs);
    /// Divide 'this' matrix by the 'rhs' scalar

/// Friend Operators

    friend ostream& operator<<(ostream& out, Matrix& other)
    {
        other.print(10);
        return out;
    }
    /// Nicely print with indentation 10
};

/*******************************************************/
///                  Creators                         ///
/*******************************************************/

template <typename T>
Matrix<T>::Matrix()
: d_rows()
{
    d_numRows = 0;
    d_numCols = 0;
}

template <typename T>
Matrix<T>::Matrix(int rows, int columns)
: d_rows(rows)
, d_numRows(rows)
, d_numCols(columns)
{
    for (int i = 0; i < rows; ++i)
        d_rows[i].resize(columns);
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
: d_rows(other.d_rows)
, d_numRows(other.d_numRows)
, d_numCols(other.d_numCols)
{
}

template <typename T>
Matrix<T>::Matrix(const vector<vector<T> >& other)
: d_rows(other)
{
    d_numRows = other.size();
    if(other.size())
        d_numCols = other[0].size();
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& rhs)
{
    if (this == &rhs) return *this; /// handle self assignment
    d_rows.clear();
    d_rows.resize(rhs.d_rows.size());
    d_numRows = rhs.numRows();
    d_numCols = rhs.numCols();
    if (rhs.d_rows.size() == 0)
        return *this;   /// handle 0 rows

    for (int i = 0; i < (int)d_rows.size(); ++i)
        d_rows[i].resize(rhs.d_rows[0].size());

    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i, j, rhs.get(i,j));
    return *this;
}
/*******************************************************/
///                  End of Creators                  ///
/*******************************************************/

/*******************************************************/
///                  Manipulators                     ///
/*******************************************************/

template <typename T>
void Matrix<T>::rowScale(int i, T scalar)
{
    for (int col = 0; col < numCols(); ++col)
        set(i, col, get(i, col) * scalar);
}

template <typename T>
void Matrix<T>::colScale(int j, T scalar)
{
    for (int row = 0; row < numRows(); ++row)
        set(row, j, get(row, j) * scalar);
}

template <typename T>
void Matrix<T>::rowSwap(int i, int j)
{
    for (int col = 0; col < numCols(); ++col) {
        T aux = get(i, col);
        set(i, col, get(j, col));
        set(j, col, aux);
    }
}

template <typename T>
void Matrix<T>::colSwap(int i, int j)
{
    for (int row = 0; row < numRows(); ++row) {
        T aux = get(row, i);
        set(row, i, get(row, j));
        set(row, j, aux);
    }
}

template <typename T>
void Matrix<T>::addToRow(int i, const Matrix& other)
{
    for (int col = 0; col < numCols(); ++col)
        set(i, col, get(i,col) + other.get(0, col));
}

template <typename T>
void Matrix<T>::addToCol(int j, const Matrix& other)
{
    for (int row = 0; row < numRows(); ++row)
        set(row, j, get(row, j) + other.get(row, 0));
}

template <typename T>
void Matrix<T>::normalize()
{
    T norm = frobeniusNorm();
    assert(norm > 0);
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i, j, get(i, j) / norm);
}

template <typename T>
void Matrix<T>::setRow(int index, const Matrix<T>& other)
{
    for (int i = 0; i < other.numCols(); ++i)
        set(index, i, other.get(0, i));
}

template <typename T>
void Matrix<T>::setColumn(int index, const Matrix<T>& other)
{
    for (int i = 0; i < other.numRows(); ++i)
        set(i, index, other.get(i, 0));
}

template <typename T>
void Matrix<T>::set(int row, int col, T value)
{
    d_rows[row][col] = value;
}

template <typename T>
Matrix<T> Matrix<T>::transpose()
{
    Matrix<T> transp(numCols(), numRows());
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            transp.set(j, i, get(i, j));
    return transp;
}

template <typename T>
T Matrix<T>::frobeniusNorm()
{
    return sqrt((transpose()* (*this)).get(0, 0));
}
/*******************************************************/
///                  End of Manipulators              ///
/*******************************************************/

/*******************************************************/
///                  Accessors                        ///
/*******************************************************/

template <typename T>
const void Matrix<T>::print(int indent) const
{
    if(numRows() == 0 || numCols() == 0) {
        cout << "empty\n";
        return;
    }
    for (int i = 0; i < numRows(); ++i) {
        for (int j = 0; j < numCols(); ++j) {
            T jt = get(i, j);
            stringstream s;
            s << setprecision(indent-4) << fixed;
            s << jt;

            assert(indent - s.str().size() >= 0);
            for (int i = 0; i < (int)(indent - s.str().size()); ++i)
                cout << " ";
            cout << s.str();
        }
        cout << "\n";
    }
}

template <typename T>
const T Matrix<T>::get(int row, int col) const
{
    return d_rows[row][col];
}

template <typename T>
const int Matrix<T>::numRows() const
{
    return d_numRows;
}

template <typename T>
const int Matrix<T>::numCols() const
{
    return d_numCols;
}

template <typename T>
Matrix<T> Matrix<T>::operator[](int index) const
{
    Matrix<T> result(1, numCols());
    for (int i = 0; i < numCols(); ++i)
        result.set(0, i, get(index, i));
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator()(int upRow,
                                int downRow,
                                int leftCol,
                                int rightCol)
{
    Matrix<T> result(downRow - upRow + 1, rightCol - leftCol + 1);
    for (int i = 0; i < result.numRows(); ++i)
        for(int j = 0; j < result.numCols(); ++j)
            result.set(i, j, get(upRow + i, leftCol + j));
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator()(int index) const
{
    Matrix<T> result(numRows(), 1);
    for (int i = 0; i < numRows(); ++i)
        result.set(i, 0, get(i, index));
    return result;
}

/*******************************************************/
///               End Of Accessors                    ///
/*******************************************************/

/*******************************************************/
///                Free Operators                     ///
/*******************************************************/

template <typename T>
bool Matrix<T>::operator == (const Matrix& other)
{/// this does not take into account floating point precision!
    if(numRows() != other.numRows() || numCols() != other.numCols())
        return false;
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            if(get(i, j) != other.get(i, j))
                return false;
    return true;
}

template <typename T>
bool Matrix<T>::operator != (const Matrix& other)
{
    return !(this == other);
}

template <typename T>
Matrix<T>& Matrix<T>::operator *= (const Matrix<T>& rhs) {

    if (numRows() == 1 && numCols() == 1) { /// Treat the case when lhs is scalar
        Matrix<T> rez(rhs);
        rez *= get(0, 0);
        (*this) = rez;
        return *this;
    }

    if (rhs.numRows() == 1 && rhs.numCols() == 1) { /// Treat the case when rhs is scalar
        Matrix<T> rez(*this);
        rez *= rhs.get(0, 0);
        (*this) = rez;
        return *this;
    }

    assert(numCols() == rhs.numRows() && "matrix multiplication impossible!");

    Matrix<T> rez(numRows(), rhs.numCols());
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < rhs.numCols(); ++j)
            for (int k = 0; k < numCols(); ++k)
                rez.set(i, j, rez.get(i, j) + get(i, k) * rhs.get(k, j));

    (*this) = rez;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator *= (const T& rhs) {
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i,j, get(i,j) * rhs);
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& rhs) {
    assert(numRows() == rhs.numRows() &&
           numCols() == rhs.numCols() &&
           "matrix addition impossible!");

    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i, j, get(i, j) + rhs.get(i, j));

    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& rhs) {
    assert(numRows() == rhs.numRows() &&
           numCols() == rhs.numCols() &&
           "matrix addition impossible!");
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i, j, get(i, j) - rhs.get(i, j));

    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator /= (const T& rhs){
    for (int i = 0; i < numRows(); ++i)
        for (int j = 0; j < numCols(); ++j)
            set(i,j, get(i,j) / rhs);
    return *this;
}

/*******************************************************/
///              End of Free Operators                ///
/*******************************************************/

/*******************************************************/
///                 Free Functions                    ///
/*******************************************************/

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> aux(lhs);
    aux *= rhs;
    return aux;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& lhs, const T& rhs) {
    Matrix<T> aux(lhs);
    aux *= rhs;
    return aux;
}

template <typename T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> aux(lhs);
    aux += rhs;
    return aux;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> aux(lhs);
    aux -= rhs;
    return aux;
}

template <typename T>
Matrix<T> operator/(const Matrix<T>& lhs, const T& rhs) {
    Matrix<T> aux(lhs);
    aux /= rhs;
    return aux;
}

/*******************************************************/
///              End of Free Functions                ///
/*******************************************************/

#endif /// MATRIX_H
