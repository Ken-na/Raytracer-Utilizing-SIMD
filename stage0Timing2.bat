@ECHO OFF
set runs=%1
if "%1"=="" set runs=5
@ECHO ON
FastRelease\Stage0_SIMDSpheres.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 4 -input Scenes/cornell.txt  
FastRelease\Stage0_SIMDSpheres.exe -runs %runs% -blockSize 16 -size 1000 1000 -samples 4 -input Scenes/allmaterials.txt 
FastRelease\Stage0_SIMDSpheres.exe -runs %runs% -blockSize 16 -size 1280  720 -samples 1 -input Scenes/5000spheres.txt 
FastRelease\Stage0_SIMDSpheres.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 1 -input Scenes/donuts.txt 
FastRelease\Stage0_SIMDSpheres.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 1 -input Scenes/cornell-199lights.txt
