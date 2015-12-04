#! /bin/bash

nvcc `pkg-config --cflags opencv` ProyectoCUDA.cu `pkg-config --libs opencv` -o proyecto

