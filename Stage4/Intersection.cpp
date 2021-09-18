/*  The following code is a VERY heavily modified from code originally sourced from:
	Ray tracing tutorial of http://www.codermind.com/articles/Raytracer-in-C++-Introduction-What-is-ray-tracing.html
	It is free to use for educational purpose and cannot be redistributed outside of the tutorial pages. */

#include "Intersection.h"
#include <immintrin.h>
#include "PrimitivesSIMD.h"


	// helper function to find "horizontal" minimum (and corresponding index value from another vector)
__forceinline void selectMinimumAndIndex(const __m256 values, const __m256i indexes, float* min, unsigned int* index)
{
	// find min of elements 1&2, 3&4, 5&6, and 7&8
	__m256 minNeighbours = _mm256_min_ps(values, _mm256_permute_ps(values, 0x31));
	// find min of min(1,2)&min(5,6) and min(3,4)&min(7,8)
	__m256 minNeighbours2 = _mm256_min_ps(minNeighbours, _mm256_permute2f128_ps(minNeighbours, minNeighbours, 0x05));
	// find final minimum 
	__m256 mins = _mm256_min_ps(minNeighbours2, _mm256_permute_ps(minNeighbours2, 0x02));

	// find all elements that match our minimum
	__m256i matchingTs = _mm256_castps_si256(_mm256_set1_ps(mins.m256_f32[0]) != values);
	// set all other elements to be MAX_INT (-1 but unsigned)
	__m256i matchingIndexes = matchingTs | indexes;

	// find minimum of remaining indexes (so smallest index will be chosen) using that same technique as above but with heaps of ugly casts
	__m256i minIndexNeighbours = _mm256_min_epu32(matchingIndexes, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(matchingIndexes), 0x31)));
	__m256i minIndexNeighbours2 = _mm256_min_epu32(minIndexNeighbours, _mm256_castps_si256(_mm256_permute2f128_ps(
		_mm256_castsi256_ps(minIndexNeighbours), _mm256_castsi256_ps(minIndexNeighbours), 0x05)));
	__m256i minIndex = _mm256_min_epu32(minIndexNeighbours2, _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(minIndexNeighbours2), 0x02)));

	// "return" minimum and associated index through reference parameters
	*min = mins.m256_f32[0];
	*index = minIndex.m256i_i32[0];
}


// test to see if collision between ray and a plane happens before time t (equivalent to distance)
// updates closest collision time (/distance) if collision occurs
// see: http://en.wikipedia.org/wiki/Line-sphere_intersection
// see: http://www.codermind.com/articles/Raytracer-in-C++-Part-I-First-rays.html
// see: Step 8 of http://meatfighter.com/juggler/ 
// this code make heavy use of constant term removal due to ray always being a unit vector
bool isSphereIntersected(const Scene* scene, const Ray* r, float* t, unsigned int* index)
{
	float tInitial = *t;

	// ray start and direction
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);

	// constants
	const __m256 epsilons = _mm256_set1_ps(EPSILON);
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256i eights = _mm256_set1_epi32(8);

	// best ts found so far and associated sphere indexes
	__m256 ts = _mm256_set1_ps(tInitial);
	__m256i indexes = _mm256_set1_epi32(*index);

	// current corresponding index
	__m256i ijs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	// search for sphere collisions, storing closest one found
	for (unsigned int i = 0; i < scene->numSpheresSIMD; ++i)
	{
		//Sphere& sphere = scene.sphereContainer[i * 8 + j];
		Vector8 pos(scene->spherePosX[i], scene->spherePosY[i], scene->spherePosZ[i]);
		__m256 sizes = scene->sphereSize[i];

		// Vector dist = pos - r.start;
		Vector8 dist = pos - rStart;

		// float B = r.dir * dist;
		__m256 Bs = dot(rDir, dist);

		// float D = B * B - dist * dist + size * size;
		__m256 Ds = Bs * Bs - dot(dist, dist) + sizes * sizes;

		// if D < 0, no intersection, so don't try and calculate the point of intersection
		//if (D < 0.0f) continue;
		__m256 DLessThanZeros = Ds < zeros;

		// calculate both intersection times(/distances)
		//float t0 = B - sqrtf(D);
		//float t1 = B + sqrtf(D);
		__m256 sqrtDs = _mm256_sqrt_ps(Ds);
		__m256 t0s = Bs - sqrtDs;
		__m256 t1s = Bs + sqrtDs;

		// check to see if either of the two sphere collision points are closer than time parameter
		//if ((t1 > EPSILON) && (t1 < t))
		__m256 t1GreaterThanEpsilonAndSmallerThanTs = (t1s > epsilons) & (t1s < ts);
		//else if ((t0 > EPSILON) && (t0 < t))
		__m256 t0GreaterThanEpsilonAndSmallerThanTs = (t0s > epsilons) & (t0s < ts);

		// select best ts 
		__m256 temp = select(t1GreaterThanEpsilonAndSmallerThanTs, t1s, ts);
		__m256 temp2 = select(t0GreaterThanEpsilonAndSmallerThanTs, t0s, temp);
		ts = select(DLessThanZeros, ts, temp2);

		// select best corresponding sphere indexes
		__m256i temp3 = select(_mm256_castps_si256(t1GreaterThanEpsilonAndSmallerThanTs), ijs, indexes);
		__m256i temp4 = select(_mm256_castps_si256(t0GreaterThanEpsilonAndSmallerThanTs), ijs, temp3);
		indexes = select(_mm256_castps_si256(DLessThanZeros), indexes, temp4);

		// increase the index counters
		ijs = _mm256_add_epi32(ijs, eights);
	}

	// extract the best t and corresponding sphere index
	selectMinimumAndIndex(ts, indexes, t, index);

	return *t < tInitial;
}


// short-circuiting version of sphere intersection test that only returns true/false
bool isSphereIntersected(const Scene* scene, const Ray* r, const float t)
{
	// ray start and direction
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);

	// constants
	const __m256 epsilons = _mm256_set1_ps(EPSILON);
	const __m256 zeros = _mm256_set1_ps(0.0f);

	// starting t
	const __m256 ts = _mm256_set1_ps(t);

	// search for sphere collisions, storing closest one found
	for (unsigned int i = 0; i < scene->numSpheresSIMD; ++i)
	{
		//Sphere& sphere = scene.sphereContainer[i * 8 + j];
		Vector8 pos(scene->spherePosX[i], scene->spherePosY[i], scene->spherePosZ[i]);
		__m256 sizes = scene->sphereSize[i];

		// Vector dist = pos - r.start;
		Vector8 dist = pos - rStart;

		// float B = r.dir * dist;
		__m256 Bs = dot(rDir, dist);

		// float D = B * B - dist * dist + size * size;
		__m256 Ds = Bs * Bs - dot(dist, dist) + sizes * sizes;

		// if D < 0, no intersection, so don't try and calculate the point of intersection
		//if (D < 0.0f) continue;
		__m256 DLessThanZeros = Ds < zeros;

		// calculate both intersection times(/distances)
		//float t0 = B - sqrtf(D);
		//float t1 = B + sqrtf(D);
		__m256 sqrtDs = _mm256_sqrt_ps(Ds);
		__m256 t0s = Bs - sqrtDs;
		__m256 t1s = Bs + sqrtDs;

		// check to see if either of the two sphere collision points are closer than time parameter
		//if ((t1 > EPSILON) && (t1 < t))
		__m256 t1GreaterThanEpsilonAndSmallerThanTs = (t1s > epsilons) & (t1s < ts);
		//else if ((t0 > EPSILON) && (t0 < t))
		__m256 t0GreaterThanEpsilonAndSmallerThanTs = (t0s > epsilons) & (t0s < ts);

		// combine all the success cases together
		__m256 success = _mm256_andnot_ps(DLessThanZeros, t0GreaterThanEpsilonAndSmallerThanTs | t1GreaterThanEpsilonAndSmallerThanTs);

		// if any are successful, short-circuit
		if (_mm256_movemask_ps(success)) return true;
	}

	return false;
}


// short-circuiting version of sphere intersection test that only returns true/false
// this version doesn't use helper functions from PrimitivesSIMD.h (i.e. only uses SIMD intrinsics directly) and 
// is only included for reference (this is /much/ harder to read / reason about than the above version)
/*bool isSphereIntersected(const Scene* scene, const Ray* r, float t)
{
	// ray start and direction
	const __m256 rStartxs = _mm256_set1_ps(r->start.x);
	const __m256 rStartys = _mm256_set1_ps(r->start.y);
	const __m256 rStartzs = _mm256_set1_ps(r->start.z);
	const __m256 rDirxs = _mm256_set1_ps(r->dir.x);
	const __m256 rDirys = _mm256_set1_ps(r->dir.y);
	const __m256 rDirzs = _mm256_set1_ps(r->dir.z);

	// constants
	const __m256 epsilons = _mm256_set1_ps(EPSILON);
	const __m256 zeros = _mm256_set1_ps(0.0f);

	// starting t
	const __m256 ts = _mm256_set1_ps(t);

	// search for sphere collisions, storing closest one found
	for (unsigned int i = 0; i < scene->numSpheresSIMD; ++i)
	{
		//Sphere& sphere = scene.sphereContainer[i * 8 + j];
		__m256 posxs = scene->spherePosX[i];
		__m256 posys = scene->spherePosY[i];
		__m256 poszs = scene->spherePosZ[i];
		__m256 sizes = scene->sphereSize[i];

		// Vector dist = pos - r.start;
		__m256 distxs = _mm256_sub_ps(posxs, rStartxs);
		__m256 distys = _mm256_sub_ps(posys, rStartys);
		__m256 distzs = _mm256_sub_ps(poszs, rStartzs);

		// float B = r.dir * dist;
		__m256 Bs = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(rDirxs, distxs), _mm256_mul_ps(rDirys, distys)), _mm256_mul_ps(rDirzs, distzs));

		// float D = B * B - dist * dist + size * size;
		__m256 distDot = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(distxs, distxs), _mm256_mul_ps(distys, distys)), _mm256_mul_ps(distzs, distzs));
		__m256 Ds = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(Bs, Bs), distDot), _mm256_mul_ps(sizes, sizes));

		// if D < 0, no intersection, so don't try and calculate the point of intersection
		//if (D < 0.0f) continue;
		__m256 DLessThanZeros = _mm256_cmp_ps(Ds, zeros, _CMP_LT_OQ);

		// calculate both intersection times(/distances)
		//float t0 = B - sqrtf(D);
		//float t1 = B + sqrtf(D);
		__m256 sqrtDs = _mm256_sqrt_ps(Ds);
		__m256 t0s = _mm256_sub_ps(Bs, sqrtDs);
		__m256 t1s = _mm256_add_ps(Bs, sqrtDs);

		// check to see if either of the two sphere collision points are closer than time parameter
		//if ((t1 > EPSILON) && (t1 < t))
		__m256 t0GreaterThanEpsilons = _mm256_cmp_ps(t0s, epsilons, _CMP_GT_OQ);
		__m256 t0SmallerThanTs = _mm256_cmp_ps(t0s, ts, _CMP_LT_OQ);
		__m256 t0GreaterThanEpsilonAndSmallerThanTs = _mm256_and_ps(t0GreaterThanEpsilons, t0SmallerThanTs);
		//else if ((t0 > EPSILON) && (t0 < t))
		__m256 t1GreaterThanEpsilons = _mm256_cmp_ps(t1s, epsilons, _CMP_GT_OQ);
		__m256 t1SmallerThanTs = _mm256_cmp_ps(t1s, ts, _CMP_LT_OQ);
		__m256 t1GreaterThanEpsilonAndSmallerThanTs = _mm256_and_ps(t1GreaterThanEpsilons, t1SmallerThanTs);

		// combine all the success cases together
		__m256 success = _mm256_andnot_ps(DLessThanZeros,
			_mm256_or_ps(t0GreaterThanEpsilonAndSmallerThanTs, t1GreaterThanEpsilonAndSmallerThanTs));

		// if any are successful, short-circuit
		if (_mm256_movemask_ps(success)) return true;
	}

	return false;
}*/


// test to see if collision between ray and a plane happens before time t (equivalent to distance)
// updates closest collision time (/distance) if collision occurs
// see: http://en.wikipedia.org/wiki/Line-plane_intersection
// see: http://www.cs.princeton.edu/courses/archive/fall00/cs426/lectures/raycast/sld017.htm
// see: http://softsurfer.com/Archive/algorithm_0104/algorithm_0104B.htm#Line-Plane Intersection
//old version
/*bool isPlaneIntersected(const Plane* p, const Ray* r, float* t)
{
	// angle between ray and surface normal
	float angle = r->dir * p->normal;

	// no intersection if ray and plane are parallel
	if (angle == 0.0f) return false;

	// find point of intersection
	float t0 = ((p->pos - r->start) * p->normal) / angle;

	// check to see if plane collision point is closer than time parameter
	if (t0 > EPSILON && t0 < *t)
	{
		*t = t0;
		return true;
	}

	return false;
}*/

bool isPlaneIntersected(const Scene* scene, const Ray* r, float* t, unsigned int* planeIndex)
{
	float startingT = *t;

	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 epsilons = _mm256_set1_ps(EPSILON);

	__m256 ts = _mm256_set1_ps(startingT);
	__m256i planeIndexes = _mm256_set1_epi32(*planeIndex);

	// ray start and direction
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);

	const __m256i eights = _mm256_set1_epi32(8);
	__m256i ijs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); //suss

	for (int i = 0; i < scene->numPlanesSIMD; i++) {
		//Vector8 normals(scene->planeContainerAoS.normal[i].x, scene->planeContainerAoS.normal[i].y, scene->planeContainerAoS.normal[i].z);
		Vector8 normals(scene->planeNormalX[i], scene->planeNormalY[i], scene->planeNormalZ[i]);
		//Vector8 poses(scene->planeContainerAoS.pos[i].x, scene->planeContainerAoS.pos[i].y, scene->planeContainerAoS.pos[i].z);
		Vector8 poses(scene->planePosX[i], scene->planePosY[i], scene->planePosZ[i]);

		__m256 angles = dot(rDir, normals);

		__m256 t0s = dot((poses - rStart), normals) / angles;

		//else if ((t0 > EPSILON) && (t0 < t))
		__m256 t0GreaterThanEpsilonAndSmallerThanTs = (t0s > epsilons) & (t0s < ts);

		__m256 succ = _mm256_and_ps(t0GreaterThanEpsilonAndSmallerThanTs, angles != zeros);

		planeIndexes = select(_mm256_castps_si256(t0GreaterThanEpsilonAndSmallerThanTs), ijs, planeIndexes);
		
		ts = select(t0GreaterThanEpsilonAndSmallerThanTs, t0s, ts);
		//planeIndexes = select(succ, planeIndexes, );

		//if (_mm256_movemask_ps(succ)) return true;
		ijs = _mm256_add_epi32(ijs, eights); //suss
	}

	selectMinimumAndIndex(ts, planeIndexes, t, planeIndex);

	return *t < startingT;
	
}

//part 3A. 256 SIMD
bool isPlaneIntersected(const Scene* scene, const Ray* r, const float t)
{
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 epsilons = _mm256_set1_ps(EPSILON);

	const __m256 ts = _mm256_set1_ps(t);

	// ray start and direction
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);
	
	//bool didHit = false;
	//float optimal = INFINITY;
	for (int i = 0; i < scene->numPlanesSIMD; i++) {
		//Vector8 normals(scene->planeContainerAoS.normal[i].x, scene->planeContainerAoS.normal[i].y, scene->planeContainerAoS.normal[i].z);
		Vector8 normals(scene->planeNormalX[i], scene->planeNormalY[i], scene->planeNormalZ[i]);
		//Vector8 poses(scene->planeContainerAoS.pos[i].x, scene->planeContainerAoS.pos[i].y, scene->planeContainerAoS.pos[i].z);
		Vector8 poses(scene->planePosX[i], scene->planePosY[i], scene->planePosZ[i]);

		__m256 angles = dot(rDir, normals);

		__m256 t0s = dot((poses - rStart), normals) / angles;

		//else if ((t0 > EPSILON) && (t0 < t))
		__m256 t0GreaterThanEpsilonAndSmallerThanTs = (t0s > epsilons) & (t0s < ts);



		__m256 succ = _mm256_and_ps(t0GreaterThanEpsilonAndSmallerThanTs, angles != zeros);

		if (_mm256_movemask_ps(succ)) return true;
	}

	return false;
}


// test to see if collision between ray and a cylinder happens before time t (equivalent to distance)
// updates closest collision time (/distance) and normal (at point of collions) if collision occurs
// based on code from: https://www.shadertoy.com/view/4lcSRn
// see: https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
// see: https://math.stackexchange.com/questions/3248356/calculating-ray-cylinder-intersection-points
// see: https://www.doc.ic.ac.uk/~dfg/graphics/graphics2010/GraphicsLecture11.pdf
bool isCylinderIntersected(const Scene* scene, const Ray* r, float* t, Vector* normal, unsigned int* planeIndex)
{
	//printf("g\n");
	bool didHit = false;
	float tInitial = *t;
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);
	const __m256 minusOnes = _mm256_set1_ps(-1.0f);
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 onesp1 = _mm256_set1_ps(1.0f);
	//const Vector8 ones(onesp1, onesp1, onesp1);
	const __m256 epsilons = _mm256_set1_ps(EPSILON);
	__m256 ts = _mm256_set1_ps(tInitial);

	__m256i indexes = _mm256_set1_epi32(*planeIndex);
	// current corresponding index
	__m256i ijs = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	Vector8 normals(normal->x, normal->y, normal->z);

	const __m256i eights = _mm256_set1_epi32(8);

	for (int i = 0; i < scene->numCylSIMD; i++) {
		// vector between start and end of the cylinder (cylinder axis, i.e. ca)
		//Vector ca = scene->cylContainerAoS.p2[i] - scene->cylContainerAoS.p1[i];

		Vector8 p1(scene->cylPos1X[i], scene->cylPos1Y[i], scene->cylPos1Z[i]);
		Vector8 p2(scene->cylPos2X[i], scene->cylPos2Y[i], scene->cylPos2Z[i]);
		__m256 sizes = _mm256_setr_ps(scene->cylContainerAoS.size[ijs.m256i_i32[0]], scene->cylContainerAoS.size[ijs.m256i_i32[1]], scene->cylContainerAoS.size[ijs.m256i_i32[2]], scene->cylContainerAoS.size[ijs.m256i_i32[3]], scene->cylContainerAoS.size[ijs.m256i_i32[4]], scene->cylContainerAoS.size[ijs.m256i_i32[5]], scene->cylContainerAoS.size[ijs.m256i_i32[6]], scene->cylContainerAoS.size[ijs.m256i_i32[7]]);
		//__m256 sizes = _mm256_set1_ps(1);

		// vector between start and end of the cylinder (cylinder axis, i.e. ca)
		//Vector ca = scene->cylContainerAoS.p2[i] - scene->cylContainerAoS.p1[i];
		Vector8 ca = p2 - p1;

		// vector between ray origin and start of the cylinder
		//Vector oc = r->start - scene->cylContainerAoS.p1[i];
		// vector between ray origin and start of the cylinder
		//Vector oc = r->start - scene->cylContainerAoS.p1[i];
		Vector8 oc = rStart - p1;


		// cache some dot-products 
		//float caca = ca * ca;
		//float card = ca * r->dir;
		//float caoc = ca * oc;
		__m256 caca = dot(ca, ca);
		__m256 card = dot(ca, rDir);
		__m256 caoc = dot(ca, oc);

		// calculate values for coefficients of line-cylinder equation
		//float a = caca - card * card;
		//float b = caca * (oc * r->dir) - caoc * card;
		//float c = caca * (oc * oc) - caoc * caoc - scene->cylContainerAoS.size[i] * scene->cylContainerAoS.size[i] * caca;
		__m256 a = caca - card * card;
		__m256 bPreCalc = dot(oc, rDir);
		__m256 b = caca * bPreCalc - caoc * card;
		__m256 cPreCalc = dot(oc, oc);
		__m256 c = caca * cPreCalc - caoc * caoc - scene->cylSize[i] * scene->cylSize[i] * caca;

		// first half of distance calculation (distance squared)
		//float h = b * b - a * c;
		__m256 h = b * b - a * c;

		// if ray doesn't intersect with infinite cylinder, exit
		//if (h < 0.0f) return false;

		// second half of distance calculation (distance)
		//h = sqrt(h);
		h = _mm256_sqrt_ps(h);

		// calculate point of intersection (on infinite cylinder)
		//float tBody = (-b - h) / a;
		__m256 tBodyPreCalc = b * minusOnes;
		__m256 tBody = (tBodyPreCalc - h) / a;

		// calculate distance along cylinder
		//float y = caoc + tBody * card;
		__m256 y = caoc + tBody * card;

		__m256 yGreaterThanZeroAndSmallerThanCaca = (y > zeros) & (y < caca);

		__m256 tBodyGreaterThanEpsilonAndSmallerThanT = (tBody > epsilons) & (tBody < ts);

		__m256 success = _mm256_and_ps(yGreaterThanZeroAndSmallerThanCaca, tBodyGreaterThanEpsilonAndSmallerThanT);

		ts = select(success, tBody, ts);
		indexes = select(_mm256_castps_si256(success), ijs, indexes);
		normals = select(success, (oc + (rDir * tBody - ca * y / caca)) / sizes, normals);

		/*for (int j = 0; j < 8; j++) {
			if (y.m256_f32[j] > 0 && y.m256_f32[j] < caca.m256_f32[j])
			{
				// check to see if the collision point on the cylinder body is closer than the time parameter
				if (tBody.m256_f32[j] > epsilons.m256_f32[j] && tBody.m256_f32[j] < *t)
				{
					*t = tBody.m256_f32[j];
					//*normal = (dot(oc, ones).m256_f32[j] + (dot(rDir, ones).m256_f32[j] * tBody.m256_f32[j] - dot(ca, ones).m256_f32[j] * y.m256_f32[j] / caca.m256_f32[j])) / scene->cylContainerAoS.size[i];
					//*normal = (oc + (r->dir * tBody.m256_f32[j] - Vector(ca.xs.m256_f32[j], ca.ys.m256_f32[j], ca.zs.m256_f32[j]) * y / caca)) / scene->cylContainerAoS.size[i];
					*planeIndex = i;
					//return true;
					didHit = true;
				}
			}
		}*/
		//Vector8 normal = (oc + (dot(rDir, tBody) - ca * y / caca)) / scene->cylContainerAoS.size[i];
		
		//indexes = _mm256_castps_si256(select(success, _mm256_castsi256_ps(indexes), ts));

		//__m256i temp4 = select(_mm256_castps_si256(success), ijs, temp3);

		// check intersection point is on the length of the cylinder
		/*if (y > 0 && y < caca)
		{
			// check to see if the collision point on the cylinder body is closer than the time parameter
			if (tBody > EPSILON && tBody < *t)
			{
				*t = tBody;
				*normal = (oc + (r->dir * tBody - ca * y / caca)) / scene->cylContainerAoS.size[i];
				*planeIndex = i;
				//return true;
				didHit = true;
			}
		}*/

		// calculate point of intersection on plane containing cap
		//float tCaps = (((y < 0.0f) ? 0.0f : caca) - caoc) / card;
		//__m256 tCaps = select(_mm256_cmp_ps(y, zeros, _CMP_LT_OQ), zeros, caca);
		__m256 tCaps = select(y < zeros, zeros, caca);
		tCaps = _mm256_sub_ps(tCaps, caoc);
		tCaps = _mm256_div_ps(tCaps, card);

		//__m256 loopCond = abs(b + a * tCaps);

		__m256 success2 = (abs(b + a * tCaps) < h) & (tCaps > epsilons & tCaps < ts);

		ts = select(success2, tCaps, ts);
		indexes = select(_mm256_castps_si256(success2), ijs, indexes);

		//__m256 signLessThan = y;

		normals = select(success2, ca * _mm256_invsqrt_ps(caca) * sign(y), normals);
		//normals = ca * invsqrtf(caca) * sign(y);

		// calculate point of intersection on plane containing cap
		/*float tCaps = (((y < 0.0f) ? 0.0f : caca) - caoc) / card;

		// check intersection point is within the radius of the cap
		if (abs(b + a * tCaps) < h)
		{
			// check to see if the collision point on the cylinder cap is closer than the time parameter
			if (tCaps > EPSILON && tCaps < *t)
			{
				*t = tCaps;
				*normal = ca * invsqrtf(caca) * sign(y);
				*planeIndex = i;
				//return true;
				didHit = true;
			}
		}*/

		ijs = _mm256_add_epi32(ijs, eights);
	}

	selectMinimumAndIndex(ts, indexes, t, planeIndex);


	/*float* xn = 0;
	float* xy = 0;
	float* xz = 0;
	selectMinimumAndIndex(normals.xs, indexes, xn, planeIndex);
	selectMinimumAndIndex(normals.ys, indexes, xy, planeIndex);
	selectMinimumAndIndex(normals.zs, indexes, xz, planeIndex);*/

	//selectMinimumAndIndex(normals, indexes, normal, planeIndex);

	//Vector newNormal = { normals.xs.m256_f32[*planeIndex], normals.ys.m256_f32[*planeIndex], normals.zs.m256_f32[*planeIndex] };
	Vector joe = { normals.xs.m256_f32[*planeIndex % 8], normals.ys.m256_f32[*planeIndex % 8], normals.zs.m256_f32[*planeIndex % 8] };
	//printf("A %f, %f, %f\n", normals.xs.m256_f32[minDex], normals.ys.m256_f32[minDex], normals.zs.m256_f32[minDex]);
	//printf("B %f, %f, %f\n", joe.x, joe.y, joe.z);


	*normal = joe;
	/*if (*t < tInitial) {
		//printf("%d", *planeIndex);
		float min = INFINITY;
		int minDex = -1;
		for (int i = 0; i < 8; i++) {
			if (ts.m256_f32[i] < min) {
				min = ts.m256_f32[i];
				minDex = i;
			}

			
			//printf("A(%d) %f, %f, %f\n", i, normals.xs.m256_f32[i], normals.ys.m256_f32[i], normals.zs.m256_f32[i]);

		}
		printf("%d, %u\n",minDex, *planeIndex);
		Vector joe = { normals.xs.m256_f32[minDex], normals.ys.m256_f32[minDex], normals.zs.m256_f32[minDex] };
		//printf("A %f, %f, %f\n", normals.xs.m256_f32[minDex], normals.ys.m256_f32[minDex], normals.zs.m256_f32[minDex]);
		//printf("B %f, %f, %f\n", joe.x, joe.y, joe.z);

		
		*normal = joe;
		//*normal = { normals.xs.m256_f32[*planeIndex % 8], normals.ys.m256_f32[*planeIndex % 8], normals.zs.m256_f32[*planeIndex % 8] };
		
		//works for all but donut.
		//*normal = { normals.xs.m256_f32[*planeIndex], normals.ys.m256_f32[*planeIndex], normals.zs.m256_f32[*planeIndex] };

	}*/
	

	return *t < tInitial;
}

//short cuircuiting / part a
bool isCylinderIntersected(const Scene* scene, const Ray* r, const float t, Vector* normal)
{
	Vector8 rStart(r->start.x, r->start.y, r->start.z);
	Vector8 rDir(r->dir.x, r->dir.y, r->dir.z);
	const __m256 minusOnes = _mm256_set1_ps(-1.0f);
	const __m256 zeros = _mm256_set1_ps(0.0f);
	const __m256 epsilons = _mm256_set1_ps(EPSILON);
	const __m256 ts = _mm256_set1_ps(t);

	//bool didHit = false;
	for (int i = 0; i < scene->numCylSIMD; i++) {
		Vector8 p1(scene->cylPos1X[i], scene->cylPos1Y[i], scene->cylPos1Z[i]);
		Vector8 p2(scene->cylPos2X[i], scene->cylPos2Y[i], scene->cylPos2Z[i]);
		// vector between start and end of the cylinder (cylinder axis, i.e. ca)
		//Vector ca = scene->cylContainerAoS.p2[i] - scene->cylContainerAoS.p1[i];
		Vector8 ca = p2 - p1;

		// vector between ray origin and start of the cylinder
		//Vector oc = r->start - scene->cylContainerAoS.p1[i];
		Vector8 oc = rStart - p1;

		// cache some dot-products 
		//float caca = ca * ca;
		//float card = ca * r->dir;
		//float caoc = ca * oc;
		__m256 caca = dot(ca, ca);
		__m256 card = dot(ca, rDir);
		__m256 caoc = dot(ca, oc);
		
		// calculate values for coefficients of line-cylinder equation
		//float a = caca - card * card;
		//float b = caca * (oc * r->dir) - caoc * card;
		//float c = caca * (oc * oc) - caoc * caoc - scene->cylContainerAoS.size[i] * scene->cylContainerAoS.size[i] * caca;
		__m256 a = caca - card * card;
		__m256 bPreCalc = dot(oc, rDir);
		__m256 b = caca * bPreCalc - caoc * card;
		__m256 cPreCalc = dot(oc, oc);
		__m256 c = caca * cPreCalc - caoc * caoc - scene->cylSize[i] * scene->cylSize[i] * caca;

		// first half of distance calculation (distance squared)
		//float h = b * b - a * c;
		__m256 h = b * b - a * c;

		// if ray doesn't intersect with infinite cylinder, exit
		//if (h < 0.0f) return false;

		// second half of distance calculation (distance)
		//h = sqrt(h);
		h = _mm256_sqrt_ps(h);

		// calculate point of intersection (on infinite cylinder)
		//float tBody = (-b - h) / a;
		__m256 tBodyPreCalc = b * minusOnes;
		__m256 tBody = (tBodyPreCalc - h) / a;

		// calculate distance along cylinder
		//float y = caoc + tBody * card;
		__m256 y = caoc + tBody * card;

		__m256 yGreaterThanZeroAndSmallerThanCaca = (y > zeros) & (y < caca);

		__m256 tBodyGreaterThanEpsilonAndSmallerThanT = (tBody > epsilons) & (tBody < ts);

		__m256 success = _mm256_and_ps(yGreaterThanZeroAndSmallerThanCaca, tBodyGreaterThanEpsilonAndSmallerThanT);
		
		if (_mm256_movemask_ps(success)) return true;
		/*
		// check intersection point is on the length of the cylinder
		if (y > 0 && y < caca)
		{
			// check to see if the collision point on the cylinder body is closer than the time parameter
			if (tBody > EPSILON && tBody < t)
			{
				//*t = tBody;
				//*normal = (oc + (r->dir * tBody - ca * y / caca)) / scene->cylContainerAoS.size[i];
				return true;
				//didHit = true;
			}
		}*/

		// calculate point of intersection on plane containing cap
		//float tCaps = (((y < 0.0f) ? 0.0f : caca) - caoc) / card;
		//__m256 tCaps = select(_mm256_cmp_ps(y, zeros, _CMP_LT_OQ), zeros, caca);
		__m256 tCaps = select(y < zeros, zeros, caca);
		tCaps = _mm256_sub_ps(tCaps, caoc);
		tCaps = _mm256_div_ps(tCaps, card);

		//__m256 loopCond = abs(b + a * tCaps);

		__m256 success2 = (abs(b + a * tCaps) < h) & (tCaps > epsilons & tCaps < ts);

		if (_mm256_movemask_ps(success2)) return true;
		/*
		// check intersection point is within the radius of the cap
		if (abs(b + a * tCaps) < h)
		{
			// check to see if the collision point on the cylinder cap is closer than the time parameter
			if (tCaps > EPSILON && tCaps < t)
			{
				//*t = tCaps;
				//*normal = ca * invsqrtf(caca) * sign(y);
				return true;
				//didHit = true;
			}
		}*/
	}

	return false;
}


// calculate collision normal, viewProjection, object's material, and test to see if inside collision object
void calculateIntersectionResponse(const Scene* scene, const Ray* viewRay, Intersection* intersect)
{
	switch (intersect->objectType)
	{
	case Intersection::PrimitiveType::SPHERE:
		intersect->normal = normalise(intersect->pos - intersect->sphere->pos);
		intersect->material = &scene->materialContainer[intersect->sphere->materialId];
		break;
	case Intersection::PrimitiveType::PLANE:
		intersect->normal = intersect->plane->normal;
		intersect->material = &scene->materialContainer[intersect->plane->materialId];
		break;
	case Intersection::PrimitiveType::CYLINDER:
		// normal already returned from intersection function, so nothing to do here
		intersect->material = &scene->materialContainer[intersect->cylinder->materialId];
		//printf("Mat-ID: %d", intersect->cylinder->materialId);
		break;
	}

	// calculate view projection
	intersect->viewProjection = viewRay->dir * intersect->normal; 

	// detect if we are inside an object (needed for refraction)
	intersect->insideObject = (intersect->normal * viewRay->dir > 0.0f);

	// if inside an object, reverse the normal
    if (intersect->insideObject)
    {
        intersect->normal = intersect->normal * -1.0f;
    }
}


// test to see if collision between ray and any object in the scene
// updates intersection structure if collision occurs
bool objectIntersection(const Scene* scene, const Ray* viewRay, Intersection* intersect)
{
	//printf("h");
	// set default distance to be a long long way away
    float t = MAX_RAY_DISTANCE;

	// no intersection found by default
	intersect->objectType = Intersection::PrimitiveType::NONE;

	// search for sphere collisions, storing closest one found
	unsigned int index = -1;
	if (isSphereIntersected(scene, viewRay, &t, &index))
	{
		intersect->objectType = Intersection::PrimitiveType::SPHERE;
		intersect->sphere = &scene->sphereContainer[index];
	}

	unsigned int planeIndex = -1;
	// search for plane collisions, storing closest one found
	if (isPlaneIntersected(scene, viewRay, &t, &planeIndex)) {
		intersect->objectType = Intersection::PrimitiveType::PLANE;
		intersect->plane = &scene->planeContainer[planeIndex];
	}

	// search for cylinder collisions, storing closest one found (and the normal at that point)
	Vector normal;
	unsigned int cylinderIndex = -1;
	if (isCylinderIntersected(scene, viewRay, &t, &normal, &cylinderIndex)) {
	//if (isCylinderIntersected(scene, )) {
		intersect->objectType = Intersection::PrimitiveType::CYLINDER;
		intersect->normal = normal;
		//printf("r: %d, %d", scene->numCylinders, cylinderIndex);
		intersect->cylinder = &scene->cylinderContainer[cylinderIndex];
	}

	// nothing detected, return false
	if (intersect->objectType == Intersection::PrimitiveType::NONE)
	{
		return false;
	}

	// calculate the point of the intersection
	intersect->pos = viewRay->start + viewRay->dir * t;

	return true;
}
