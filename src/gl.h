#ifndef __GL_H__
#define __GL_H__

#include "tgaimage.h"
#include "Matrix.h"

extern Matrix4f MODEL;
extern Matrix4f VIEW;
extern Matrix4f PROJECTION; 
extern Matrix4f VIEWPORT;

struct IShader{
    virtual ~IShader();
    virtual Vec4f vertex(int iface, int nthvert) = 0;
    virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float *zbuffer);

#endif //__GL_H__

