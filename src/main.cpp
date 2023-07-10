#include "Model.h"
#include "Matrix.h"
#include "tgaimage.h"
#include "gl.h"

#include <iostream>

const int width = 1024;
const int height = 1024;

Model *model = nullptr;

Vec3f light_pos(1,1,1);
Vec3f light_dir(2,2,2);
Vec3f       eye(1,1,3);
// Vec3f       eye(1.5f,2,3);
Vec3f    center(0,0,0);
Vec3f        up(0,1,0);

struct FlatShader : public IShader{
    Matrix<float, 2, 3>uvs;
    Matrix<float, 3, 3>vertex_pos;
    virtual Vec4f vertex(int iface, int nthvert){
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert));
        uvs.SetCol(nthvert, model->uv(iface, nthvert));
        vertex_pos.SetCol(nthvert, to_vec3(gl_Vertex));
        return VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex; 
    }
    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec3f e1 = vertex_pos.Col(1) - vertex_pos.Col(0);
        Vec3f e2 = vertex_pos.Col(2) - vertex_pos.Col(0);
        Vec3f face_normal = normalize(e1.cross(e2));
        Vec2f uv = uvs * bc_coords;
        TGAColor diffuse = model->diffuse(uv);
        float intensity = std::max(0.f, face_normal.dot(light_dir));
        color = diffuse*intensity;
    }
};

struct GouraudShader : public IShader {
    Vec3f varying_intensity; // written by vertex shader, read by fragment shader

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert).dot(light_dir)); // get diffuse lighting intensity
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert)); // read the vertex from .obj file
        // transform it to screen coordinates
        return VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex; 
    }

    virtual bool fragment(Vec3f bc_coords, TGAColor &color) {
        float intensity = varying_intensity.dot(bc_coords);   // interpolate intensity for the current pixel
        if (intensity>.85) intensity = 1;
        else if (intensity>.60) intensity = .80;
        else if (intensity>.45) intensity = .60;
        else if (intensity>.30) intensity = .45;
        else if (intensity>.15) intensity = .30;
        else intensity = 0;
        color = TGAColor(255, 155, 0)*intensity;
        return false;                              // no, we do not discard this pixel
    }
};

struct TexShader : public IShader{
    Vec3f varying_intensity;
    Vec2f uvs[3]; 

    virtual Vec4f vertex(int iface, int nthvert){
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert).dot(light_dir));
        uvs[nthvert] = model->uv(iface, nthvert);
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert));
        return VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex; 
    }

    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        float intensity = varying_intensity.dot(bc_coords); 
        Vec2f uv = bc_coords[0]*uvs[0] + bc_coords[1]*uvs[1] + bc_coords[2]*uvs[2];
        color = model->diffuse(uv);
        return false;
    }
};

struct NormalShader : public IShader{
    Vec2f uvs[3];
    virtual Vec4f vertex(int iface, int nthevert){
        uvs[nthevert] = model->uv(iface, nthevert);
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthevert));
        return VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex; 
    }
    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec2f uv = bc_coords[0]*uvs[0] + bc_coords[1]*uvs[1] + bc_coords[2]*uvs[2];
        Vec3f normal = model->normal(uv);
        // transform normal from [-1,1]->[0,255]
        for(int i=0;i<3;i++)color[i] = 255.f * (normal[i]/2 + 0.5f);
        return false;
    }
};

// Phong Shading & Blinn-Phone Shading
struct PhongShader : public IShader{
    Mat4f uniform_M;
    Mat4f uniform_MIT;
    Vec2f uvs[3];
    Mat3f varying_tri;
    
    PhongShader(Mat4f M, Mat4f MIT):uniform_M(M), uniform_MIT(MIT){}
    
    virtual Vec4f vertex(int iface, int nthvert){
        uvs[nthvert] = model->uv(iface, nthvert);
        Vec4f gl_Vertex = VIEWPORT*PROJECTION*VIEW*MODEL*to_vec4(model->vert(iface, nthvert));
        varying_tri.SetCol(nthvert, to_vec3(gl_Vertex/gl_Vertex.w));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec2f uv = bc_coords[0]*uvs[0] + bc_coords[1]*uvs[1] + bc_coords[2]*uvs[2];
        Vec3f n = to_vec3(uniform_MIT*to_vec4(model->normal(uv))).normalize();
        /* parallel light */
        Vec3f l = to_vec3(uniform_M*to_vec4(light_dir)).normalize();
        
        /* point light */
        // Vec3f frag_pos = varying_tri * bc_coords;
        // Vec3f l = to_vec3(uniform_MIT*to_vec4(light_pos - frag_pos)).normalize();
        
        float diff = std::max(0.f, n.dot(l));
        /* 
            Phong Shading
                n: normal 
                l: light direction 
                r: reflect light direction
        */

        Vec3f r = (n.dot(l) * 2.0f * n - l).normalize();
        float spec = pow(std::max(normalize(eye).dot(r), 0.0f), model->specular(uv));
        
        /* 
            Blinn-Phong Shading
                n: normal 
                l: light direction
                h: half vector
        */
        // Vec3f h = (normalize(eye)+l).normalize();
        // float spec = pow(std::max(n.dot(h), 0.0f), model->specular(uv));
        
        TGAColor c = model->diffuse(uv);
        for(int i=0;i<3;i++) color[i] = std::min(5 + c[i]*(diff + 0.6f * spec), 255.0f);
        return false;
    } 
};

// tangent normal mapping
struct TBNShader : public IShader{
    Matrix<float, 2, 3> varying_uv;
    Matrix<float, 3, 3> varying_normal;
    Matrix<float, 3, 3> varying_vertex;
    Matrix<float, 3, 3> varying_tangent;
    
    virtual Vec4f vertex(int iface, int nthvert){
        varying_uv.SetCol(nthvert, model->uv(iface, nthvert));
        varying_normal.SetCol(nthvert, model->normal(iface, nthvert));
        varying_vertex.SetCol(nthvert, model->vert(iface, nthvert));
        varying_tangent.SetCol(nthvert, model->tangent(iface, nthvert));
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert));
        return VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex; 
    }
    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec2f uv = varying_uv * bc_coords;
        Vec3f normal = model->normal(uv);
        // calculating TBN Matrix
        // if((e1.cross(e2)).dot(normal)>0)printf("counter-clockwise\n");else printf("counter-clockwise\n"); 
        Vec3f world_space_normal = to_vec3( (MODEL).inverse().tranpose() * to_vec4(varying_normal*bc_coords)).normalize();
        Vec3f world_space_tangent = to_vec3( MODEL * to_vec4(varying_tangent*bc_coords)).normalize(); 
        Vec3f Normal =  world_space_normal.normalize();
        Vec3f Tangent = normalize(world_space_tangent - world_space_tangent.dot(world_space_normal)*world_space_normal);
        Vec3f Bitangnet = vector_cross(Normal, Tangent).normalize();
        Mat3f tangentToWorld( {Tangent, Bitangnet, Normal} );
        tangentToWorld = tangentToWorld.tranpose(); // !!!!! 必须要重新赋值 不然会出现左半边好的，右半边坏的情况
        normal = normalize(tangentToWorld*normal);
        Vec3f n = normal;
        Vec3f l = to_vec3( MODEL * to_vec4(light_dir)).normalize();
        float diffuse = std::max(0.f, n.dot(l));
        Vec3f r = (n.dot(l) * 2.0f * n - l).normalize();
        float specular = pow(std::max(normalize(eye).dot(r), 0.0f), model->specular(uv));
        TGAColor c = model->diffuse(uv);
        for(int i=0;i<3;i++)color[i] = std::min(5 + c[i]*(diffuse + 0.6f * specular), 255.0f);
        // for(int i=0;i<3;i++)color[i] = 255.f * (normal[i]/2 + 0.5f);
        return false;
    }    
};

struct DepthShader : public IShader{
    Mat3f varying_tri;
    DepthShader() : varying_tri() {}
    virtual Vec4f vertex(int iface, int nthvert){
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert));
        gl_Vertex = VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex;
        varying_tri.SetCol(nthvert, to_vec3(gl_Vertex/gl_Vertex.w));
        return gl_Vertex;
    }
    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec3f p = varying_tri * bc_coords;
        color = TGAColor(255, 255, 255)*(1.0f - (p.z/2 + 0.5f));
        return false;
    }
};

struct ShadowShader : public IShader{
    Mat4f uniform_M;
    Mat4f uniform_MIT;
    Mat4f uniform_MShadow;
    Mat3f varying_tri;
    Matrix<float, 2, 3> varying_uv;
    float *depth_buffer;

    ShadowShader(Mat4f M, Mat4f MIT, Mat4f MShadow, float *depth_buffer):
    uniform_M(M), uniform_MIT(MIT), uniform_MShadow(MShadow), depth_buffer(depth_buffer){}
    virtual Vec4f vertex(int iface, int nthvert){
        Vec4f gl_Vertex = to_vec4(model->vert(iface, nthvert));
        gl_Vertex = VIEWPORT*PROJECTION*VIEW*MODEL*gl_Vertex;
        varying_tri.SetCol(nthvert, to_vec3(gl_Vertex/gl_Vertex.w));
        varying_uv.SetCol(nthvert, model->uv(iface, nthvert));
        return gl_Vertex;
    }
    virtual bool fragment(Vec3f bc_coords, TGAColor &color){
        Vec4f shadow_p = uniform_MShadow * to_vec4(varying_tri * bc_coords);
        shadow_p = shadow_p/shadow_p.w;
        int idx = int(shadow_p[0]) + int(shadow_p[1])*width;
        float shadow_coef = depth_buffer[idx]  < (shadow_p.z/2+0.5f - 0.04f) ? 0.3f : 1.0f;
        Vec2f uv = varying_uv * bc_coords;
        Vec3f n = to_vec3(uniform_MIT*to_vec4(model->normal(uv))).normalize();
        Vec3f l = to_vec3(uniform_M*to_vec4(light_dir)).normalize();
        Vec3f r = (n.dot(l) * 2.0f * n - l).normalize();
        float spec = pow(std::max(normalize(eye).dot(r), 0.0f), model->specular(uv));
        float diff = std::max(0.f, n.dot(l));
        TGAColor c = model->diffuse(uv);
        for(int i=0;i<3;i++) color[i] = std::min(20 + c[i]*shadow_coef*(1.2f*diff + 0.6f*spec), 255.0f);
        return false;
    }
};

int main(int argc, char **argv)
{
    if(argc == 2)
        model = new Model(argv[1]);
    else 
        model = new Model("Resources/african_head.obj");

    TGAImage framebuffer(width, height, TGAImage::RGB);
    TGAImage depthmap(width, height, TGAImage::GRAYSCALE);

    MODEL = set_model_matrix();
    VIEW = set_view_matrix(light_dir, center, up);
    PROJECTION = set_perspective_matrix(45.f, width/height, -1.0f, -5.f);
    VIEWPORT = set_viewport_matrix(width, height);
    light_dir.normalize();

    // FlatShader shader;
    // GouraudShader shader;
    // TexShader shader;    
    // NormalShader shader;
    // PhongShader shader;

    // buggy: tangent normal mapping
    // TBNShader shader;


    float *depth_buffer = new float[width*height]; for(int i=0;i<width*height;i++) depth_buffer[i]=1.0f;
    DepthShader depth_shader;
    float maxD = -999.0f, minD = 999.f;
    for(int i = 0; i < model->nfaces(); i++){
        Vec4f screen_coords[3];
        for(int j = 0; j < 3; j++){
            screen_coords[j] = depth_shader.vertex(i,j);
            float d = screen_coords[j].z / screen_coords[j].w;
            if(d > maxD) maxD = d;
            if(d < minD) minD = d;
        }
        triangle(screen_coords, depth_shader, depthmap, depth_buffer);
    }
    printf("max depth: %f, min depth: %f\n", maxD, minD);
    depthmap.write_tga_file("depthmap.tga");

    Mat4f M_Depth = VIEWPORT*PROJECTION*VIEW*MODEL; 

    MODEL = set_model_matrix();
    VIEW = set_view_matrix(eye, center, up);
    PROJECTION = set_perspective_matrix(45.f, width/height, -2.0f, -5.f);
    VIEWPORT = set_viewport_matrix(width, height);
    float *zbuffer = new float[width*height]; for(int i=0;i<width*height;i++) zbuffer[i]=1.0f;
    
    PhongShader shader(MODEL, (MODEL).inverse().tranpose());
    // TexShader shader;
    // NormalShader shader;
    // TBNShader shader;
    // ShadowShader shader(MODEL, (MODEL).inverse().tranpose(), M_Depth*((VIEWPORT*PROJECTION*VIEW*MODEL).inverse()), depth_buffer);

    for(int i = 0; i < model->nfaces(); i++){
        Vec4f screen_coords[3];
        for(int j = 0; j < 3; j++){
            screen_coords[j] = shader.vertex(i,j);
        }
        triangle(screen_coords, shader, framebuffer, zbuffer);
    }
    framebuffer.write_tga_file("image.tga");

    delete model;
    return 0;
}