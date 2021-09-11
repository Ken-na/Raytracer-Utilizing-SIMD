/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#ifndef __INTERSECTION_H
#define __INTERSECTION_H

#include "Scene.h"
#include "SceneObjects.h"

	// all pertinant information about an intersection of a ray with an object
typedef struct Intersection
{
	enum class PrimitiveType { NONE, SPHERE, PLANE, CYLINDER };

	PrimitiveType objectType;	// type of object intersected with

	Point pos;											// point of intersection
	Vector normal;										// normal at point of intersection
	float viewProjection;								// view projection 
	bool insideObject;									// whether or not inside an object

	Material* material;									// material of object

	// object collided with
	union
	{
		Sphere* sphere;
		Cylinder* cylinder;
		Plane* plane;
	};
} Intersection;

// test to see if collision between ray and a plane happens before time t (equivalent to distance)
// updates closest collision time (/distance) if collision occurs
bool isSphereIntersected(const Scene* scene, const Ray* r, float* t, unsigned int* index);

// short circuiting version of the above
bool isSphereIntersected(const Scene* scene, const Ray* r, const float t);

// test to see if collision between ray and a plane happens before time t (equivalent to distance)
// updates closest collision time (/distance) if collision occurs
bool isPlaneIntersected(const Scene* scene, const Ray* r, float* t, unsigned int* planeIndex);

//short circuiting ver
bool isPlaneIntersected(const Scene* scene, const Ray* r, const float t);

// test to see if collision between ray and a cylinder happens before time t (equivalent to distance)
// updates closest collision time (/distance) if collision occurs
bool isCylinderIntersected(const Cylinder* c, const Ray* r, float* t, Vector* normal);

// calculate collision normal, viewProjection, object's material, and test to see if inside collision object
void calculateIntersectionResponse(const Scene* scene, const Ray* viewRay, Intersection* intersect);

// test to see if collision between ray and any object in the scene
// updates intersection structure if collision occurs
bool objectIntersection(const Scene* scene, const Ray* viewRay, Intersection* intersect);

#endif // __INTERSECTION_H
