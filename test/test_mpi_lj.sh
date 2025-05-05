test_path=$(pwd)

foldername="tmp_$(date +%Y%m%d_%H%M%S)"
mkdir temp/${foldername}
cp test_mpi_lj.sh temp/${foldername}/test_mpi_lj.sh 

cd temp
cd ${foldername}


echo ${test_path}
mpirun -np 6 python ${test_path}/../ns_run < ${test_path}/data/inputs.test.periodic.GMC.fortran
# /home/nico.unglert/code/pymatnest/ns_run < data/inputs.test.periodic.GMC.fortran > profile.out