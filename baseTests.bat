@rem doskey magick = c:\Program Files\ImageMagick-7.0.10-Q8\magick.exe

Release\RayTracerAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_01.bmp -input Scenes/cornell.txt           
Release\RayTracerAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_02.bmp -input Scenes/allmaterials.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_03.bmp -input Scenes/5000spheres.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_04.bmp -input Scenes/donuts.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_05.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\base_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\basediff_01.bmp
magick compare -metric mae Outputs\base_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\basediff_02.bmp
magick compare -metric mae Outputs\base_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\basediff_03.bmp
magick compare -metric mae Outputs\base_04.bmp Outputs_REFERENCE\stage0_04.bmp Outputs\basediff_04.bmp
magick compare -metric mae Outputs\base_05.bmp Outputs_REFERENCE\stage0_05.bmp Outputs\basediff_05.bmp

Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_06.bmp -input Scenes/cornell.txt           
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_07.bmp -input Scenes/allmaterials.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_08.bmp -input Scenes/5000spheres.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_09.bmp -input Scenes/donuts.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_10.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\base_06.bmp Outputs_REFERENCE\stage0_06.bmp Outputs\basediff_06.bmp
magick compare -metric mae Outputs\base_07.bmp Outputs_REFERENCE\stage0_07.bmp Outputs\basediff_07.bmp
magick compare -metric mae Outputs\base_08.bmp Outputs_REFERENCE\stage0_08.bmp Outputs\basediff_08.bmp
magick compare -metric mae Outputs\base_09.bmp Outputs_REFERENCE\stage0_09.bmp Outputs\basediff_09.bmp
magick compare -metric mae Outputs\base_10.bmp Outputs_REFERENCE\stage0_10.bmp Outputs\basediff_10.bmp

Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_11.bmp -input Scenes/cornell.txt           
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_12.bmp -input Scenes/allmaterials.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_13.bmp -input Scenes/5000spheres.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_14.bmp -input Scenes/donuts.txt 
Release\RayTracerAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_15.bmp -input Scenes/cornell-199lights.txt
                                                                                
magick compare -metric mae Outputs\base_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\basediff_11.bmp
magick compare -metric mae Outputs\base_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\basediff_12.bmp
magick compare -metric mae Outputs\base_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\basediff_13.bmp
magick compare -metric mae Outputs\base_14.bmp Outputs_REFERENCE\stage0_14.bmp Outputs\basediff_14.bmp
magick compare -metric mae Outputs\base_15.bmp Outputs_REFERENCE\stage0_15.bmp Outputs\basediff_15.bmp
