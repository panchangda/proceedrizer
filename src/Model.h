#pragma once
#ifndef __MODEL_H__
#define __MODEL_H__

#include "Matrix.h"
#include "tgaimage.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

class Model
{
public:
    Model() {}
    Model(const char *filename){
        std::string file_prefix = filename;
        int dot_pos = file_prefix.find_last_of('.');
        file_prefix = file_prefix.substr(0, dot_pos);
        LoadFromObj(filename);
        CalculateTangetSpace();
        LoadDiffuse( (file_prefix + "_diffuse.tga").c_str());
        LoadNormal( (file_prefix + "_nm.tga").c_str());
        // LoadNormal( (file_prefix + "_nm_tangent.tga").c_str());
        LoadSpecular( (file_prefix + "_spec.tga").c_str());
    }
    ~Model(){
        if(_diffusemap) delete _diffusemap;
        if(_specularmap) delete _specularmap;
        if(_normalmap) delete _normalmap;
    }

    int nverts() const { return _verts.size(); }
    int nfaces() const { return _faces_verts.size(); }

    Vec3f vert(int i) const { return _verts[i]; }
    Vec3f vert(int iface, int nthvert) const { return _verts[_faces_verts[iface][nthvert]]; }
    Vec3f normal(int iface, int nthvert) const { return normalize(_norms[_faces_norms[iface][nthvert]]); }
    Vec2f uv(int iface, int nthvert) const { return _uv[_faces_uv[iface][nthvert]]; }
    Vec3f tangent(int iface, int nthvert) const { return _tangents[_faces_tangents[iface][nthvert]]; }

    TGAColor diffuse(Vec2f uv) const 
    {
        if(!_diffusemap) 
            return TGAColor(255,255,255); 
        else 
            return _diffusemap->get(uv.x *_diffuse_width, (1-uv.y) *_diffuse_height);     
    }

    Vec3f normal(Vec2f uv) const
    {
        if(!_normalmap)
            return Vec3f();
        else{
            TGAColor normal_color = _normalmap->get(uv.x *_normal_width, (1.f-uv.y) *_normal_height);
            Vec3f res;
            // transform [0,255] -> [0,1] -> [-1,1]
            // switch x&z coords
            // for(int i=0;i<3;i++)res[2-i] = normal_color[i]/255.f * 2.0f - 1.0f;
            for(int i=0;i<3;i++)res[i] = normal_color[i]/255.f * 2.0f - 1.0f;
            return res;
        }
    } 

    float specular(Vec2f uv) const
    {
        if(!_specularmap)
            return 0.f;
        else 
        {
            TGAColor specular_color = _specularmap->get(uv.x *_specular_width, (1-uv.y) *_specular_height); 
            return specular_color[0]/1.f; 
        }
    }

    void LoadFromObj(const char *filename)
    {
        std::ifstream in;
        in.open(filename, std::ifstream::in);
        if (in.fail())
            return;

        std::string line;
        while (!in.eof())
        {
            std::getline(in, line);
            std::istringstream iss(line.c_str());
            char trash;
            if (line.compare(0, 2, "v ") == 0)
            {
                iss >> trash;
                Vec3f v;
                for (int i = 0; i < 3; i++)
                    iss >> v[i];
                this->_verts.push_back(v);
            }
            else if (line.compare(0, 3, "vn ") == 0)
            {
                iss >> trash >> trash;
                Vec3f vn;
                for (int i = 0; i < 3; i++)
                    iss >> vn[i];
                _norms.push_back(vn);
            }
            else if (line.compare(0, 3, "vt ") == 0)
            {
                iss >> trash >> trash;
                Vec2f uv;
                iss >> uv[0] >> uv[1];
                _uv.push_back(uv);
            }
            else if (line.compare(0, 2, "f ") == 0)
            {
                iss >> trash;
                std::vector<int> v;
                std::vector<int> vt;
                std::vector<int> vn;
                int v_idx, vt_idx, vn_idx;
                while (iss >> v_idx >> trash >> vt_idx >> trash >> vn_idx)
                {
                    v.push_back(v_idx - 1);
                    vt.push_back(vt_idx - 1);
                    vn.push_back(vn_idx - 1);
                }
                _faces_verts.push_back(v);
                _faces_uv.push_back(vt);
                _faces_norms.push_back(vn);
            }
        }

        std::cerr << "# v# " << _verts.size() << " f# " << _faces_verts.size() << std::endl;
    }

    void LoadDiffuse(const char *filename)
    {
        TGAImage* diffusemap = new TGAImage();
        bool isReadSuccess = diffusemap->read_tga_file(filename);
        if (isReadSuccess){
            std::cout << "Diffuse Loaded!" << std::endl;
            _diffusemap = diffusemap;
            _diffuse_height = diffusemap->height();
            _diffuse_width = diffusemap->width();
        }
        else{
            std::cout << "Error: Failed to Read Diffuse from " << filename << std::endl; 
        }
    }

    void LoadNormal(const char *filename)
    {
        TGAImage* normalmap = new TGAImage();
        bool isReadSuccess = normalmap->read_tga_file(filename);
        if (isReadSuccess){
            std::cout << "Normal Loaded!" << std::endl;
            _normalmap = normalmap;
            _normal_height = normalmap->height();
            _normal_width = normalmap->width();
        }
        else{
            std::cout << "Error: Failed to Read Normal from " << filename << std::endl; 
        }
    }
    
    void LoadSpecular(const char *filename)
    {
        TGAImage* specularmap = new TGAImage();
        bool isReadSuccess = specularmap->read_tga_file(filename);
        if (isReadSuccess){
            std::cout << "Specular Loaded!" << std::endl;
            _specularmap = specularmap;
            _specular_height = specularmap->height();
            _specular_width = specularmap->width();
        }
        else{
            std::cout << "Error: Failed to Read Specular from " << filename << std::endl; 
        }
    }

    void CalculateTangetSpace(){
        _tangents.resize(this->nverts());
        _faces_tangents.resize(this->nfaces());
        for(int i=0;i<this->nfaces();i++){
            Vec3f vert[3];
            Vec2f uv[3];
            for(int j=0;j<3;j++){
                vert[j] = this->vert(i,j);
                uv[j] = this->uv(i,j);
            }
            Vec3f e1 = vert[1] - vert[0];
            Vec3f e2 = vert[2] - vert[0];
            float deltaU1 = uv[1].x - uv[0].x;
            float deltaU2 = uv[2].x - uv[0].x;
            float deltaV1 = uv[1].y - uv[0].y;
            float deltaV2 = uv[2].y - uv[0].y;
            float f = 1.0f / (deltaU1*deltaV2 - deltaU2*deltaV1);
            Vec3f tangent = f * (deltaV2*e1 - deltaV1*e2);
            std::vector<int> face_tangent(3);
            for(int j=0;j<3;j++){
                int nthvert = this->_faces_verts[i][j];
                _tangents[nthvert] = normalize(tangent);
                face_tangent[j] = nthvert;
            }
            _faces_tangents[i] = face_tangent;
        }
    }

protected:
    std::vector<Vec3f> _verts;
    std::vector<Vec2f> _uv;
    std::vector<Vec3f> _norms;
    std::vector<Vec3f> _tangents;
    std::vector<std::vector<int> > _faces_verts;
    std::vector<std::vector<int> > _faces_uv;
    std::vector<std::vector<int> > _faces_norms;
    std::vector<std::vector<int> > _faces_tangents;

    TGAImage* _diffusemap = nullptr;
    int _diffuse_height = 0;
    int _diffuse_width = 0;
    TGAImage* _specularmap = nullptr;
    int _specular_height = 0;
    int _specular_width = 0;
    TGAImage* _normalmap = nullptr;
    int _normal_height = 0;
    int _normal_width = 0;
};

#endif //__MODEL_H__