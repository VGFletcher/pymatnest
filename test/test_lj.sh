test_path=$(pwd)
cd temp
foldername="tmp_$(date +%Y%m%d_%H%M%S)"
mkdir ${foldername}
cd ${foldername}

echo ${test_path}
python ${test_path}/../ns_run < ${test_path}/data/inputs.test.periodic.GMC.fortran
# /home/nico.unglert/code/pymatnest/ns_run < data/inputs.test.periodic.GMC.fortran > profile.out