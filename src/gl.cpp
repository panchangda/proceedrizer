#include "gl.h"
#include "tgaimage.h"
#include "Matrix.h"

Matrix4f MODEL;
Matrix4f VIEW;
Matrix4f PROJECTION; 
Matrix4f VIEWPORT;

IShader::~IShader() {}


void line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color)
{
    bool steep = false;
    if (std::abs(x1 - x0) < std::abs(y1 - y0)){
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }
    if (x0 > x1){
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = std::abs(dy) * 2;
    int error2 = 0;
    int y = y0;
    int yincr = y1 > y0 ? 1 : -1;

    if (steep){
        for (int x = x0; x <= x1; x++){
            image.set(y, x, color);
            error2 += derror2;
            if (error2 >= dx){
                y += yincr;
                error2 -= dx * 2;
            }
        }
    }
    else{
        for (int x = x0; x <= x1; x++){
            image.set(x, y, color);
            error2 += derror2;
            if (error2 >= dx){
                y += yincr;
                error2 -= dx * 2;
            }
        }
    }
}
void line (Vec2i p0, Vec2i p1, TGAImage &image, TGAColor color){
    line(p0.x, p0.y, p1.x, p1.y, image, color);
}

void horizontal_line(int x0, int x1, int y, TGAImage &image, TGAColor color){
    if(x0 > x1)
        std::swap(x0, x1);
    for(int i = x0; i <= x1; i++){
        image.set(i, y, color);
    }
}
// scan-line fill
void scan_line_triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) { 

    // TGAColor white = TGAColor(255, 255, 255, 255);
    // TGAColor red = TGAColor(255, 0, 0, 255);
    // TGAColor green = TGAColor(0, 255, 0, 255);
    // TGAColor blue = TGAColor(0, 0, 255, 255);

    if (t0.y>t1.y) std::swap(t0, t1); 
    if (t0.y>t2.y) std::swap(t0, t2); 
    if (t1.y>t2.y) std::swap(t1, t2);

    int height_t0t2 = t2.y - t0.y;
    int height_t0t1 = t1.y - t0.y;
    int height_t1t2 = t2.y - t1.y;

    // upper half 
    for(int y = t0.y; y <= t1.y; y++){
        float alpha = float(y - t0.y) / height_t0t2;
        float beta = float(y- t0.y) / height_t0t1;
        int x_t0t2 = t0.x + (t2.x - t0.x)*alpha;
        int x_t0t1 = t0.x + (t1.x - t0.x)*beta;
        // image.set(x_t0t2, y, red);
        // image.set(x_t0t1, y, green); 
        horizontal_line(x_t0t2, x_t0t1, y, image, color);
    }
    //lower half
    for(int y = t1.y; y <= t2.y; y++){
        float alpha = float(y - t0.y) / height_t0t2;
        float beta = float(y - t1.y) / height_t1t2;
        int x_t0t2 = t0.x + (t2.x - t0.x)*alpha;
        int x_t1t2 = t1.x + (t2.x - t1.x)*beta;
        // image.set(x_t0t2, y, red);
        // image.set(x_t1t2, y, blue);
        horizontal_line(x_t0t2, x_t1t2, y, image, color);
    }
}

/*

P = alpha * A + beta * B + gamma * C

=> P = (1-u-v)A + uB + vC;
=> uAB + vAC + PA = 0

                    t_x
=> [u, v, 1] [AB_x, AC_x, PA_x].T = 0
                    t_y
=> [u, v, 1] [AB_y, AC_y, PA_y].T = 0

=> [u, v, 1] // t_x cross product t_y

*/

Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2i P){
    Vec3f s[2];
    for(int i = 0; i < 2; i++){
        s[i][0] = B[i]-A[i];
        s[i][1] = C[i]-A[i];
        s[i][2] = A[i]-P[i];
    }
    Vec3f uv1 = s[0].cross(s[1]);
    // if uv1.z < 0
    if(std::abs(uv1.z) < 1e-2) return Vec3f(-1, 1, 1);
    return Vec3f(1.f - (uv1.x+uv1.y)/uv1.z, uv1.x/uv1.z, uv1.y/uv1.z);
}

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float* zbuffer){
    // found bbox
    Vec2f bboxmin(image.width()-1, image.height()-1);
    Vec2f bboxmax(0, 0);
    Vec2f image_boundry(image.width()-1,image.height()-1);
    for(int i = 0; i < 3; i++){
        bboxmin.x = std::max(0.f, std::min( pts[i].x/pts[i].w, bboxmin.x));
        bboxmin.y = std::max(0.f, std::min( pts[i].y/pts[i].w, bboxmin.y));
        bboxmax.x = std::min(image_boundry.x, std::max( pts[i].x/pts[i].w, bboxmax.x));
        bboxmax.y = std::min(image_boundry.y, std::max( pts[i].y/pts[i].w, bboxmax.y));
    }
    // shading pixels
    Vec2i P(0, 0);
    for(P.x = bboxmin.x; P.x <= bboxmax.x; P.x++){
        for(P.y = bboxmin.y; P.y <= bboxmax.y; P.y++){
            Vec3f bc_coords = barycentric((pts[0]/pts[0].w).xy(), (pts[1]/pts[1].w).xy(), (pts[2]/pts[2].w).xy(), P);
            if(bc_coords.x<0 || bc_coords.y<0 || bc_coords.z<0) continue;
            float fragment_depth = bc_coords.x * pts[0].z/pts[0].w + bc_coords.y * pts[1].z/pts[1].w + bc_coords.z * pts[2].z/pts[2].w;            
            // depth-testing: transform fragment-z(screen space [-1,1]) to [0,1]
            if(fragment_depth/2+0.5f >= zbuffer[P.x+P.y*image.width()] ) continue;


            // fragment shading
            TGAColor color;
            bool discard = shader.fragment(bc_coords, color);


            // fill z-buffer
            if(!discard){
                zbuffer[P.x+P.y*image.width()] = fragment_depth/2+0.5f;  // update z-buffer
                image.set(P.x, P.y, color); // replace pixel color
            }
        }
    }
}

