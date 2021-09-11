@rem doskey magick = c:\Program Files\ImageMagick-7.0.10-Q8\magick.exe

Release\Stage4.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage4_01.bmp -input Scenes/cornell.txt           
Release\Stage4.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage4_02.bmp -input Scenes/allmaterials.txt 
Release\Stage4.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage4_03.bmp -input Scenes/5000spheres.txt 
Release\Stage4.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage4_04.bmp -input Scenes/donuts.txt 
Release\Stage4.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage4_05.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\stage4_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\stage4diff_01.bmp
magick compare -metric mae Outputs\stage4_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\stage4diff_02.bmp
magick compare -metric mae Outputs\stage4_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\stage4diff_03.bmp
magick compare -metric mae Outputs\stage4_04.bmp Outputs_REFERENCE\stage0_04.bmp Outputs\stage4diff_04.bmp
magick compare -metric mae Outputs\stage4_05.bmp Outputs_REFERENCE\stage0_05.bmp Outputs\stage4diff_05.bmp

Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage4_06.bmp -input Scenes/cornell.txt           
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage4_07.bmp -input Scenes/allmaterials.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage4_08.bmp -input Scenes/5000spheres.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage4_09.bmp -input Scenes/donuts.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage4_10.bmp -input Scenes/cornell-199lights.txt

magick compare -metric mae Outputs\stage4_06.bmp Outputs_REFERENCE\stage0_06.bmp Outputs\stage4diff_06.bmp
magick compare -metric mae Outputs\stage4_07.bmp Outputs_REFERENCE\stage0_07.bmp Outputs\stage4diff_07.bmp
magick compare -metric mae Outputs\stage4_08.bmp Outputs_REFERENCE\stage0_08.bmp Outputs\stage4diff_08.bmp
magick compare -metric mae Outputs\stage4_09.bmp Outputs_REFERENCE\stage0_09.bmp Outputs\stage4diff_09.bmp
magick compare -metric mae Outputs\stage4_10.bmp Outputs_REFERENCE\stage0_10.bmp Outputs\stage4diff_10.bmp

Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage4_11.bmp -input Scenes/cornell.txt           
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage4_12.bmp -input Scenes/allmaterials.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage4_13.bmp -input Scenes/5000spheres.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage4_14.bmp -input Scenes/donuts.txt 
Release\Stage4.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage4_15.bmp -input Scenes/cornell-199lights.txt
                                                                                
magick compare -metric mae Outputs\stage4_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\stage4diff_11.bmp
magick compare -metric mae Outputs\stage4_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\stage4diff_12.bmp
magick compare -metric mae Outputs\stage4_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\stage4diff_13.bmp
magick compare -metric mae Outputs\stage4_14.bmp Outputs_REFERENCE\stage0_14.bmp Outputs\stage4diff_14.bmp
magick compare -metric mae Outputs\stage4_15.bmp Outputs_REFERENCE\stage0_15.bmp Outputs\stage4diff_15.bmp
