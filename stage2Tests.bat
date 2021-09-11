@rem doskey magick = c:\Program Files\ImageMagick-7.0.10-Q8\magick.exe

Release\Stage2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage2_01.bmp -input Scenes/cornell.txt           
Release\Stage2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage2_02.bmp -input Scenes/allmaterials.txt 
Release\Stage2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage2_03.bmp -input Scenes/5000spheres.txt 
Release\Stage2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage2_04.bmp -input Scenes/donuts.txt 
Release\Stage2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage2_05.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\stage2_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\stage2diff_01.bmp
magick compare -metric mae Outputs\stage2_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\stage2diff_02.bmp
magick compare -metric mae Outputs\stage2_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\stage2diff_03.bmp
magick compare -metric mae Outputs\stage2_04.bmp Outputs_REFERENCE\stage0_04.bmp Outputs\stage2diff_04.bmp
magick compare -metric mae Outputs\stage2_05.bmp Outputs_REFERENCE\stage0_05.bmp Outputs\stage2diff_05.bmp

Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage2_06.bmp -input Scenes/cornell.txt           
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage2_07.bmp -input Scenes/allmaterials.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage2_08.bmp -input Scenes/5000spheres.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage2_09.bmp -input Scenes/donuts.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage2_10.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\stage2_06.bmp Outputs_REFERENCE\stage0_06.bmp Outputs\stage2diff_06.bmp
magick compare -metric mae Outputs\stage2_07.bmp Outputs_REFERENCE\stage0_07.bmp Outputs\stage2diff_07.bmp
magick compare -metric mae Outputs\stage2_08.bmp Outputs_REFERENCE\stage0_08.bmp Outputs\stage2diff_08.bmp
magick compare -metric mae Outputs\stage2_09.bmp Outputs_REFERENCE\stage0_09.bmp Outputs\stage2diff_09.bmp
magick compare -metric mae Outputs\stage2_10.bmp Outputs_REFERENCE\stage0_10.bmp Outputs\stage2diff_10.bmp

Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage2_11.bmp -input Scenes/cornell.txt           
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage2_12.bmp -input Scenes/allmaterials.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage2_13.bmp -input Scenes/5000spheres.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage2_14.bmp -input Scenes/donuts.txt 
Release\Stage2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage2_15.bmp -input Scenes/cornell-199lights.txt
                                                                                
magick compare -metric mae Outputs\stage2_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\stage2diff_11.bmp
magick compare -metric mae Outputs\stage2_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\stage2diff_12.bmp
magick compare -metric mae Outputs\stage2_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\stage2diff_13.bmp
magick compare -metric mae Outputs\stage2_14.bmp Outputs_REFERENCE\stage0_14.bmp Outputs\stage2diff_14.bmp
magick compare -metric mae Outputs\stage2_15.bmp Outputs_REFERENCE\stage0_15.bmp Outputs\stage2diff_15.bmp
