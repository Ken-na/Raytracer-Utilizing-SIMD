/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

// YOU SHOULD _NOT_ NEED TO MODIFY THIS FILE

#ifndef __SCENE_H
#define __SCENE_H

#include "SceneObjects.h"
#include <immintrin.h>

// description of a single static scene
typedef struct Scene 
{
	Point cameraPosition;					// camera location
	float cameraRotation;					// direction camera points
    float cameraFieldOfView;				// field of view for the camera

	float exposure;							// image exposure

	unsigned int skyboxMaterialId;

	// scene object counts
	unsigned int numMaterials;
	unsigned int numLights;
	unsigned int numSpheres;
	unsigned int numPlanes;
	unsigned int numCylinders;

	// scene objects
	Material* materialContainer;	
	Light* lightContainer;
	Sphere* sphereContainer;
	Plane* planeContainer;
	Cylinder* cylinderContainer;

	// SIMD spheres
	unsigned int numSpheresSIMD;
	__m256* spherePosX;
	__m256* spherePosY;
	__m256* spherePosZ;
	__m256* sphereSize;
	__m256i* sphereMaterialId;
} Scene;

bool init(const char* inputName, Scene& scene);

#endif // __SCENE_H
