#pragma once
#ifndef __Matrix_H__
#define __Matrix_H__

#include <cassert>
#include <cmath>
#include <iostream>

/*
    modifed from: 
        1. RenderHelp.h from skywind3000
        2. tinyrenderer from ssloy

    design:
        1. struct 只能在内部定义函数 xx.yy() 导致必须特化某些模板，代码冗余
        用class重构后能否在外部定义特化函数如：className::yy 来减少代码冗余？
        2. why struct based?

    features:
        1. eigen-like api
        2. template based 
        3. struct based
*/


/* Vector */

template <typename T, size_t N> struct Vector{
    T m[N];
    inline Vector(){ for(size_t i = 0; i < N; i++) m[i] = T();}
    inline Vector(const Vector<T, N> &u) { for(size_t i = 0; i < N; i++) m[i] = u.m[i]; }
    inline Vector(const T *ptr) { for(size_t i = 0; i < N; i++) m[i] = ptr[i]; }
    inline const T& operator [] (size_t i) const    { assert(i < N); return m[i]; }
	inline T&       operator [] (size_t i)          { assert(i < N); return m[i]; }
};

// Specialzie Template 2D Vector
template <typename T> struct Vector<T, 2>{
    union{
        struct{ T x, y; };
        struct{ T u, v; };
        T m[2];
    };
    inline Vector(): x(T()), y(T()) {}
    inline Vector(T X, T Y): x(X), y(Y) {}
    inline Vector(const Vector<T, 2> &u): x(u.x), y(u.y) {}
    inline Vector(const T *ptr): x(ptr[0]), y(ptr[1]) {}
    inline const T& operator [] (size_t i) const    { assert( i < 2); return m[i]; }
    inline T&       operator [] (size_t i)          { assert( i < 2); return m[i]; }
    inline T _x() const { return x; }
    inline T _y() const { return y; }  
    inline Vector<T, 2> xy() const              { return Vector<T, 2>(x,y); }
    inline Vector<T, 3> xyz(T z=1) const        { return Vector<T, 3>(x,y,z); }
    inline Vector<T, 4> xyzw(T z=1, T w=1) const{ return Vector<T, 4>(x,y,z,w); }
};

// Specialzie Template 3D Vector
template <typename T> struct Vector<T, 3>{
    union{
        struct{ T x, y, z; };
        struct{ T r, g, b; };
        struct{ T ivert, iuv, inorm; };
        T m[3];
    };
    inline Vector<T, 3>(): x(T()), y(T()), z(T()) {}
    inline Vector<T, 3>(T X, T Y, T Z): x(X), y(Y), z(Z) {}
    inline Vector<T, 3>(const Vector<T, 3> &u): x(u.x), y(u.y), z(u.z) {}
    inline Vector<T, 3>(const T *ptr): x(ptr[0]), y(ptr[1]), z(ptr[2]) {}
    inline const T& operator [] (size_t i) const    { assert( i < 3); return m[i]; }
    inline T&       operator [] (size_t i)          { assert( i < 3); return m[i]; }
    inline T _x() const { return x; }
    inline T _y() const { return y; }
    inline T _z() const { return z; }
    inline Vector<T, 2> xy() const          { return Vector<T, 2>(x,y); }
    inline Vector<T, 3> xyz() const         { return Vector<T, 3>(x,y,z); }
    inline Vector<T, 4> xyzw(T w=1) const   { return Vector<T, 4>(x,y,z,w); }

    // special for Vector<T, 3>
    inline float norm() const { return std::sqrt(x*x + y*y + z*z); }
    inline Vector<T, 3>& normalize() { *this = (*this)/this->norm(); return *this; }
    inline float dot(const Vector<T, 3> &u) const {return x*u.x + y*u.y + z*u.z; }
    inline Vector<T, 3> cross (const Vector<T, 3> &v) const {return Vector<T, 3>(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x);}
    inline Vector<T, 3> operator ^ (const Vector<T, 3> &v) const { return Vector<T, 3>(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x); }
    
};

// Specialize Template for 4D Vector
template <typename T> struct Vector<T, 4>{
    union{
        struct{ T x, y, z, w; };
        struct{ T r, g, b, a; };
        T m[4];
    };
    inline Vector<T, 4>(): x(T()), y(T()), z(T()), w(T()) {}
    inline Vector<T, 4>(T X, T Y, T Z, T W): x(X), y(Y), z(Z), w(W) {}
    inline Vector<T, 4>(const Vector<T, 4> &u): x(u.x), y(u.y), z(u.z), w(u.w) {}
    inline Vector<T, 4>(const T *ptr): x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}
    inline const T& operator [] (size_t i) const    { assert( i < 4); return m[i]; }
    inline T&       operator [] (size_t i)          { assert( i < 4); return m[i]; }
    inline T _x() const { return x; }
    inline T _y() const { return y; }
    inline T _z() const { return z; }
    inline T _w() const { return w; }
    inline Vector<T, 2> xy() const          { return Vector<T, 2>(x,y); }
    inline Vector<T, 3> xyz() const         { return Vector<T, 3>(x,y,z); }
    inline Vector<T, 4> xyzw() const        { return Vector<T, 4>(x,y,z,w); }
};

// = -a
template <typename T, size_t N> 
inline Vector<T, N> operator - (const Vector<T, N> &a) {
	Vector<T, N> b;
	for (size_t i = 0; i < N; i++) b[i] = -a[i];
	return b;
}

// = a + b
template <typename T, size_t N>
inline Vector<T, N> operator + (const Vector<T, N> &a, const Vector<T, N> &b){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] + b[i];
    return c;
}

// = a - b
template <typename T, size_t N>
inline Vector<T, N> operator - (const Vector<T, N> &a, const Vector<T, N> &b){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] - b[i];
    return c;
}

// = a * b
template <typename T, size_t N>
inline Vector<T, N> operator * (const Vector<T, N> &a, const Vector<T, N> &b){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] * b[i];
    return c;
}

// = a * x
template <typename T, size_t N>
inline Vector<T, N> operator * (const Vector<T, N> &a, T x){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] * x;
    return c;
}

// = x * a
template <typename T, size_t N>
inline Vector<T, N> operator * (T x, const Vector<T, N> &a){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] * x;
    return c;
}

// = a / x
template <typename T, size_t N>
inline Vector<T, N> operator / (const Vector<T, N> &a, T x){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = a[i] / x;
    return c;
}

// = x / a
template <typename T, size_t N>
inline Vector<T, N> operator / (T x, const Vector<T, N> &a){
    Vector<T, N> c;
    for (size_t i = 0; i < N; i++) c[i] = x / a[i];
    return c;
}

// = ostream << a.m
template <typename T, size_t N>
inline std::ostream& operator << (std::ostream &os, const Vector<T, N> &a){
    os << "[";
    for (size_t i = 0; i < N; i++) {
        os << a[i];
        if( i < N-1 ) os << ", ";
    }
    os << "]";
    return os;
}

// dot product
template <typename T, size_t N>
inline T vector_dot(const Vector<T, N> &a, const Vector<T, N> &b){
    T sum = 0;
    for (size_t i = 0; i < N; i++) sum += a[i] * b[i];
    return sum; 
}

// 2d cross product => scalar
template<typename T>
inline T vector_cross(const Vector<T, 2>& a, const Vector<T, 2>& b) {
	return a.x * b.y - a.y * b.x;
}

// 3d cross product => 3d vector
template<typename T>
inline Vector<T, 3> vector_cross(const Vector<T, 3>& a, const Vector<T, 3>& b) {
	return Vector<T, 3>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

template<typename T>
inline Vector<T, 3> normalize(const Vector<T, 3> &a){
    Vector<T, 3> b(a);
    return b.normalize();
}

template<typename T>
inline Vector<T, 4> to_vec4(const Vector<T, 3> &a){
    return a.xyzw();
}

template<typename T>
inline Vector<T, 3> to_vec3(const Vector<T, 4> &a){
    return a.xyz();
}



/* Matrix */

template <typename T, size_t ROW, size_t COL> struct Matrix{
    T m[ROW][COL];

    inline Matrix(){}
    inline Matrix(const Matrix<T, ROW, COL> &src){
        for(size_t r = 0; r < ROW; r++){
            for(size_t c = 0; c < COL; c++)
                m[r][c] = src[r][c];
        }
    }
    inline Matrix(const std::initializer_list<Vector<T, COL>> &u){
        auto it = u.begin();
        for(size_t i = 0; i < ROW; i++) SetRow(i, *it++);
    }
    inline const T* operator [] (size_t row) const  { assert(row < ROW); return m[row]; }
    inline T*       operator [] (size_t row)        { assert(row < ROW); return m[row]; }
    inline Vector<T, COL> Row(size_t row_index) const {
        assert(row_index < ROW);
        Vector<T, COL> tmp;
        for (size_t i = 0; i < COL; i++) tmp[i] = m[row_index][i];
        return tmp; 
    }

    // get a column
    inline Vector<T, ROW> Col(size_t col_index) const {
        assert(col_index < COL);
        Vector<T, ROW> tmp;
        for (size_t i = 0; i < ROW; i++) tmp[i] = m[i][col_index];
        return tmp;
    }

    // set a row
    inline void SetRow(size_t row_index, const Vector<T, COL>& a){
        assert(row_index < ROW);
        for (size_t i = 0; i < COL; i++) { m[row_index][i] = a[i]; }
    }

    // set a column
    inline void SetCol(size_t col_index, const Vector<T, ROW>& a){
        assert(col_index < COL);
        for (size_t i = 0; i < ROW; i++) { m[i][col_index] = a[i]; }
    }
    
    // identity
    inline static Matrix<T, ROW, COL> Identity(){
        Matrix<T, ROW, COL> Mat;
        for (size_t r = 0; r < ROW; r++){
            for (size_t c = 0; c < COL; c++){
                Mat.m[r][c] = r == c ? 1 : 0;
            }
        }
        return Mat;        
    }

    // zero
    inline static Matrix<T, ROW, COL> Zero(){
        Matrix<T, ROW, COL> Mat;
        for (size_t r = 0; r < ROW; r++){
            for (size_t c = 0; c < COL; c++){
                Mat.m[r][c] = 0;
            }
        }
        return Mat;        
    }

    // minor
    inline Matrix<T, ROW-1, COL-1> Minor(size_t row_index, size_t col_index) const{
        Matrix<T, ROW-1, COL-1> Mat;
        for (size_t r = 0; r < ROW - 1; r++) {
            for (size_t c = 0; c < COL -1; c++) {
                Mat.m[r][c] = m[r < row_index ? r : r+1][ c < col_index ? c : c+1];
            }
        }
        return Mat;
    }

    // transpose
    inline Matrix<T, ROW, COL> tranpose() const{
        Matrix<T, ROW, COL> Mat;
        for (size_t r = 0; r < ROW; r++) {
            for (size_t c = 0; c < COL; c++) {
                Mat.m[c][r] = m[r][c];
            }
        }
        return Mat;
    }


};

// Specialzie Template NxN Matrix
template <typename T, size_t N> struct Matrix<T, N, N>{
    T m[N][N];
 
    inline Matrix(){}
    // initializing with vectors
    inline Matrix(const std::initializer_list<Vector<T, N>> &u){
        auto it = u.begin();
        for(size_t r = 0; r < N; r++)   SetRow(r, *it++);
    }
    inline const T* operator [] (size_t row) const  { assert(row < N); return m[row]; }
    inline T*       operator [] (size_t row)        { assert(row < N); return m[row]; }

    // get a row
    inline Vector<T, N> Row(size_t row_index) const {
        assert(row_index < N);
        Vector<T, N> tmp;
        for (size_t i = 0; i < N; i++) tmp[i] = m[row_index][i];
        return tmp; 
    }

    // get a column
    inline Vector<T, N> Col(size_t col_index) const {
        assert(col_index < N);
        Vector<T, N> tmp;
        for (size_t i = 0; i < N; i++) tmp[i] = m[i][col_index];
        return tmp;
    }

    // set a row
    inline void SetRow(size_t row_index, const Vector<T, N>& a){
        assert(row_index < N);
        for (size_t i = 0; i < N; i++) { m[row_index][i] = a[i]; }
    }

    // set a column
    inline void SetCol(size_t col_index, const Vector<T, N>& a){
        assert(col_index < N);
        for (size_t i = 0; i < N; i++) { m[i][col_index] = a[i]; }
    }

    // identity
    inline static Matrix<T, N, N> Identity(){
        Matrix<T, N, N> Mat;
        for (size_t r = 0; r < N; r++){
            for (size_t c = 0; c < N; c++){
                Mat.m[r][c] = r == c ? 1 : 0;
            }
        }
        return Mat;        
    }

    // zero
    inline static Matrix<T, N, N> Zero(){
        Matrix<T, N, N> Mat;
        for (size_t r = 0; r < N; r++){
            for (size_t c = 0; c < N; c++){
                Mat.m[r][c] = 0;
            }
        }
        return Mat;        
    }

    // minor
    inline Matrix<T, N-1, N-1> Minor(size_t row_index, size_t col_index) const{
        Matrix<T, N-1, N-1> Mat;
        for (size_t r = 0; r < N - 1; r++) {
            for (size_t c = 0; c < N - 1; c++) {
                Mat.m[r][c] = m[r < row_index ? r : r+1][ c < col_index ? c : c+1];
            }
        }
        return Mat;
    }

    // transpose
    inline Matrix<T, N, N> tranpose() const{
        Matrix<T, N, N> Mat;
        for (size_t r = 0; r < N; r++){
            for (size_t c = 0; c < N; c++){
                Mat.m[c][r] = m[r][c];
            }
        }
        return Mat;
    }

    // determinant
    inline T determinant() const{
        if constexpr (N==1) return m[0][0];
        else if constexpr (N==2) return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        else { 
            T sum = 0; 
            for (size_t i = 0; i < N; i++)  sum += m[0][i] * this->cofactor(0, i); 
            return sum;
        }
    }

    // cofactor
    inline T cofactor(size_t row_index, size_t col_index) const{
        if constexpr (N == 1) return 0;
        else return this->Minor(row_index, col_index).determinant() * (((row_index + col_index) % 2)? -1 : 1);
    }

    // adjoint
    inline Matrix<T, N, N> adjoint() const{
        Matrix<T, N, N> Mat;
        for (size_t r = 0; r < N; r++){
            for (size_t c = 0; c < N; c++){
                Mat[r][c] = this->cofactor(c, r);
            }
        }
        return Mat;
    }

    // inverse
    inline Matrix<T, N, N> inverse() const{
        Matrix<T, N, N> Mat = this->adjoint();
        T det = vector_dot(this->Row(0), Mat.Col(0));
        return Mat / det;
    }
    
    // adjugate
    inline Matrix<T, N, N> adjugate() const{
        Matrix<T, N, N> Mat;
        for(size_t r = 0; r < N; r++){
            for(size_t c = 0; c < N; c++){
                Mat[r][c] = this->cofactor(r, c);
            }
        }
        return Mat;
    }
    // invert_transpose
    inline Matrix<T, N, N> invert_transpose() const{
        Matrix<T, N, N> Mat = this->adjugate();
        T det = vector_dot(this->Row(0), Mat.Row(0));
        return Mat / det;
    }
};

/* 

    Specialize To Prevent Recursive Template Size Exceeds Error
    e.g. https://www.likecs.com/ask-318678.html
    for platform.cppstandard < c++17

*/

// template <typename T> struct Matrix<T, 1, 1>{
//     T m[1][1];

//     inline Matrix(){ **m = T();}
//     inline Matrix(T x) { **m = x; }
//     inline const T* operator [] (size_t row) const  { assert(row < 1); return m[row]; }
//     inline T*       operator [] (size_t row)        { assert(row < 1); return m[row]; }
    
//     // identity
//     inline static Matrix<T, 1, 1> Identity(){
//         return Matrix<T, 1, 1>(1);        
//     }

//     // zero
//     inline static Matrix<T, 1, 1> Zero(){
//         return Matrix<T, 1, 1>(0);       
//     }

//     // determinant
//     inline T determinant() const{
//         return m[0][0];
//     }

//     // cofactor
//     inline T cofactor(size_t row_index, size_t col_index) const{
//         return 0;
//     }

// };

// Matrix + Matrix
template <typename T, size_t ROW, size_t COL> 
inline Matrix<T, ROW, COL> operator + (const Matrix<T, ROW, COL> &a, const Matrix<T, ROW, COL> &b){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat[r][c] = a[r][c] + b[r][c];
        }
    }
    return Mat;
}

// Matrix - Matrix
template <typename T, size_t ROW, size_t COL> 
inline Matrix<T, ROW, COL> operator - (const Matrix<T, ROW, COL> &a, const Matrix<T, ROW, COL> &b){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat[r][c] = a[r][c] - b[r][c];
        }
    }
    return Mat;
}

// Matrix * Matrix
// MxP = MxN * NxP
template <typename T, size_t M, size_t N, size_t P> 
inline Matrix<T, M, P> operator * (const Matrix<T, M, N> &a, const Matrix<T, N, P> &b){
    Matrix<T, M, P> Mat;
    for (size_t r = 0; r < M; r++){
        for (size_t c = 0; c < P; c++){
            Mat.m[r][c] = vector_dot(a.Row(r), b.Col(c)); 
        }
    }
    return Mat;
} 

// Vector * Matrix
// 1xCOL = 1xROW * ROWxCOL 
template <typename T, size_t ROW, size_t COL>
inline Vector<T, COL> operator * (const Vector<T, COL> &a, const Matrix<T, ROW, COL> &b){
    Vector<T, COL> vec;
    for (size_t i = 0; i < COL; i++){
        vec.m[i] = vector_dot(a, b.Col(i));
    }
    return vec;
}

// Matrix * Vector
// ROWx1 = ROWxCOL * COLx1
template <typename T, size_t ROW, size_t COL>
inline Vector<T, ROW> operator * (const Matrix<T, ROW, COL> &a, const Vector<T, COL> &b){
    Vector<T, ROW> vec;
    for (size_t i = 0; i < ROW; i++){
        vec.m[i] = vector_dot(a.Row(i), b);
    }
    return vec;
}

// Matrix * scalar
template <typename T, size_t ROW, size_t COL>
inline Matrix<T, ROW, COL> operator * (const Matrix<T, ROW, COL> &a, const T x){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat.m[r][c] = a.m[r][c] * x;
        }
    }
    return Mat;
} 

// scalar * Matrix
template <typename T, size_t ROW, size_t COL>
inline Matrix<T, ROW, COL> operator * (const T x, const Matrix<T, ROW, COL> &a){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat.m[r][c] = a.m[r][c] * x;
        }
    }
    return Mat;
}

// Matrix / scalar
template <typename T, size_t ROW, size_t COL>
inline Matrix<T, ROW, COL> operator / (const Matrix<T, ROW, COL> &a, const T x){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat.m[r][c] = a.m[r][c] / x;
        }
    }
    return Mat;
} 

// scalar / Matrix
template <typename T, size_t ROW, size_t COL>
inline Matrix<T, ROW, COL> operator / (const T x, const Matrix<T, ROW, COL> &a){
    Matrix<T, ROW, COL> Mat;
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            Mat.m[r][c] = x / a.m[r][c];
        }
    }
    return Mat;
} 

// os << Matrix
template <typename T, size_t ROW, size_t COL>
inline std::ostream& operator << (std::ostream &os, const Matrix<T, ROW, COL> &a){
    for (size_t r = 0; r < ROW; r++){
        for (size_t c = 0; c < COL; c++){
            os << a.m[r][c];
            if(c < COL-1) os << "\t";
        }
        os << "\n";
    }
    return os;
}



// alias definition
typedef Vector<int, 2>      Vec2i;
typedef Vector<float, 2>    Vec2f;
typedef Vector<double, 2>   Vec2d;
typedef Vector<int, 3>      Vec3i;
typedef Vector<float, 3>    Vec3f;
typedef Vector<double, 3>   Vec3d;
typedef Vector<int, 4>      Vec4i;
typedef Vector<float, 4>    Vec4f;
typedef Vector<double, 4>   Vec4d;
typedef Vector<int, 2>      Vector2i;
typedef Vector<float, 2>    Vector2f;
typedef Vector<double, 2>   Vector2d;
typedef Vector<int, 3>      Vector3i;
typedef Vector<float, 3>    Vector3f;
typedef Vector<double, 3>   Vector3d;
typedef Vector<int, 4>      Vector4i;
typedef Vector<float, 4>    Vector4f;
typedef Vector<double, 4>   Vector4d;

typedef Matrix<float, 2, 2> Mat2f;
typedef Matrix<double, 2, 2>Mat2d;
typedef Matrix<float, 3, 3> Mat3f;
typedef Matrix<double, 3, 3>Mat3d;
typedef Matrix<float, 4, 4> Mat4f;
typedef Matrix<double, 4, 4>Mat4d;
typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<double, 3, 3>Matrix3d;
typedef Matrix<float, 4, 4> Matrix4f;
typedef Matrix<double, 4, 4>Matrix4d;


/* 

    0. opengl-like api
    1. right hand coordinate system 
    2. matrix left multiply (Column Order)

Object Space --(Model)--> World Space --(Camera)--> Camera Space --(Projection)--> Clip Space --(Viewport)--> Screen Space
    
Model...

View    
    |x'.x   x'.y    x'.z    x' dot -eye|
    |y'.x   y'.y    y'.z    y' dot -eye|
    |z'.x   z'.y    z'.z    z' dot -eye|
    |0      0       0       1          |

Perspetive
    |2n/(r-l)               (l+r)/(l-r) 0           |
    |           2n/(t-b)    (b+t)/(b-t) 0           |
    |                       (n+f)/(n-f) 2fn/(f-n)   |
    |                       1           0           |

r=-l && t=-b
<===========>
    |n/r                                    |
    |       n/t                             |            
    |               (n+f)/(n-f)   2fn/(f-n) | 
    |               1                       |

ViewPort
    |w/2        0       0       (w-1)/2 |
    |0          h/2 0   0       (h-1)/2 |
    |0          0       1       1       |
    |0          0       0       1       |

*/

inline static Matrix4f set_model_matrix(){
    return Matrix4f::Identity();
}



inline static Matrix4f set_view_matrix(const Vector3f &eye, const Vector3f &center, const Vector3f &up){
    Matrix4f view;
    Vector3f z = (eye-center).normalize();
    Vector3f x = vector_cross(up, z).normalize();
    Vector3f y = vector_cross(z, x).normalize();
    // std::cout << z << x << y << std::endl;
    view.SetRow(0, Vector4f(x.x,x.y,x.z,-x.dot(eye)));
    view.SetRow(1, Vector4f(y.x,y.y,y.z,-y.dot(eye)));
    view.SetRow(2, Vector4f(z.x,z.y,z.z,-z.dot(eye)));
    // view.SetRow(0, Vector4f(x.x,x.y,x.z,-center.x));
    // view.SetRow(1, Vector4f(y.x,y.y,y.z,-center.y));
    // view.SetRow(2, Vector4f(z.x,z.y,z.z,-center.z));
    view.SetRow(3, Vector4f(0.f,0.f,0.f,1.f));
    return view;
}

constexpr double PI = 3.1415926;

inline static Matrix4f set_perspective_matrix(float fovy, float aspect, float znear, float zfar){
    Matrix4f perspective = Matrix4f::Zero();
    // fax = 1/(t/n) = n/t 
    // aspect = r/t
    float fax = 1.0f / std::tan(fovy/2.0f/180*PI);
    perspective[0][0] = fax/aspect;
    perspective[1][1] = fax;
    perspective[2][2] = (znear+zfar)/(znear-zfar);
    perspective[2][3] = 2*znear*zfar/(zfar-znear);
    // 
    perspective[3][2] = -1;
    return perspective;
}

inline static Matrix4f set_viewport_matrix(float width, float height){
    Matrix4f viewport = Matrix4f::Identity();
    viewport[0][0] = width/2;
    viewport[1][1] = height/2;
    viewport[0][3] = (width-1)/2;
    viewport[1][3] = (height-1)/2;
    return viewport;
}


#endif //__Matrix_H__