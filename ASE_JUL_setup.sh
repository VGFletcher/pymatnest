#!/bin/bash

#Install script to allow the use of ACE potentials using the ASE calculator in Python
#Author V.G.Fletcher

#Instructions were written based on:
#https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/

#1) Check Julia exists and can be used
echo -e "Checking that Julia is installed and in PATH...\n"
julia --version

if [ $? -eq 0 ]
then
    echo -e "\nSuccessfully found Julia executable in PATH"
else
    echo -e "\nFailure, you need to install Julia to this machine and put the executable in the PATH"
    echo -e "Please see: https://julialang.org/downloads/"
    exit
fi

#2) Use Julia to setup the project and environment
#2a) User provide path to create project
echo -e "Now to setup the Julia project\n"
echo -e "Julia works in a similar way to Conda and Python"
echo -e "We will create a Julia project (like a Conda environment)"
echo -e "This directory should be given in your nested sampling input file through the keyword ACE_env_path"
echo -e "Example: /home/test_dir/"
read -p "Please provide a directory to store this project:" julia_env

echo -e "Provided path: $julia_env"

if [ -z $julia_env ]
then
    echo -e "No path provided, please provide a path"
fi

echo -e "Attempting to create project at given path..."

#2b) Creating julia project with required modules
julia --project=$julia_env ASE_JUL_setup.jl

if [ $? -eq 0 ]
then
    echo -e "\nSuccessfully created Julia project with correct modules"
else
    echo -e "\nFailed to create Julia with project with required modules"
    exit
fi

#3) Installing pyjulip
echo -e "Attempting to install pyjulip..."
python -m pip install julia

if [ $? -eq 0 ]
then
    echo -e "\nSuccessfully installed PyJulip"
else
    echo -e "\nFailed to install PyJulip, please see: https://github.com/casv2/pyjulip"
    exit
fi

#4) Installing Julia in Python
echo -e "Attempting to install Julia in Python"
python -c "import julia; julia.install()"

if [ $? -eq 0 ]
then
    echo -e "\nSuccessfully installed Julia for Python"
else
    echo -e "\nFailed to install Julia for Python"
    exit
fi

#5) Final setup
echo -e "Attempting final setup of PyJulip..."
git clone https://github.com/casv2/pyjulip.git
cd pyjulip
pip install .

if [ $? -eq 0 ]
then
    echo -e "\nAll commands executed successfully!"
    echo -e "\nTo use the ACE_JUL calculator in python remember to set ACE_env_path=$julia_env in your nested sampling input file"
else
    echo -e "\nFailed to setup PyJulip, please see: https://github.com/casv2/pyjulip"
    exit
fi
