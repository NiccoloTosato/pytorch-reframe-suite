# SPDX-FileCopyrightText: 2026 Niccolo Tosato niccolo.tosato@yahoo.it
#
# SPDX-License-Identifier: MIT

import os
import re
import sys
import pathlib
import reframe as rfm
import reframe.utility.sanity as sn
import urllib.request
from pytorch_test_base import PyTorchTestBase
from reframe.core.builtins import (
    fixture,
    parameter,
    performance_function,
    run_after,
    run_before,
    sanity_function,
    variable,
)
from reframe.core.meta import make_test
from packaging import version

from utils import GetLatestPytorch,GetCliVar

class VersionsHandlerPlugin(rfm.RegressionTestPlugin):
    """
    This regression test plugin is necessary to propagate variables to any test without fixtures.
    """
    rocm_version_pytorch = variable(str,value="7.2")
    rocm_version_module =  variable(str,value="7.2.0")

class PyTorchAmdTestBase(PyTorchTestBase):
    descr = 'Check the training throughput on AMD MI250x'
    valid_systems = ['*']
    throughput_per_gpu = 530
    env_vars = {
        'NCCL_SOCKET_IFNAME': 'hsn0',
        'NCCL_DEBUG': 'Info',
        'NCCL_NET_GDR_LEVEL': 3,
        'NCCL_CROSS_NIC': 1
    }

class FetchAws(rfm.RunOnlyRegressionTest,VersionsHandlerPlugin):
    descr = "Fetch aws-ofi-<X>ccl"
    # Warning, version apply only to nccl release, no rccl.
    version = variable(str,value='1.19.1')
    executable = 'git'
    local = True
    @sanity_function
    def assert_clone_succeeded(self):
        return sn.assert_eq(self.job.exitcode,0)
    @run_before('run')
    def configure_clone_source(self):
        if version.parse(self.rocm_version_module) >= version.parse("7.1.0"):
            # Get NEW nccl ofi
            self.url = f"https://github.com/aws/aws-ofi-nccl.git"
            self.executable_opts = [f"clone {self.url}"]
            self.postrun_cmds = ["cd aws-ofi-nccl", f"git checkout {self.version}", "./autogen.sh"]  
        else:
            # Fallback using the OLD aws-ofi-rccl (deprecated!)
            self.url = f"https://github.com/ROCm/aws-ofi-rccl.git"
            self.executable_opts = [f"clone {self.url}"]
            self.postrun_cmds = ["cd aws-ofi-rccl", f"git checkout cxi", "./autogen.sh"]  

class SetupAwsNccl(rfm.CompileOnlyRegressionTest,VersionsHandlerPlugin):
    descr = 'Build aws-ofi network plugin (NCCL/RCCL)'
    build_system = 'Autotools'
    compiler = variable(str,value="hipcc")
    build_prefix = variable(str)
    local = True
    aws = fixture(FetchAws,scope='session')
    @run_before('compile')
    def configure_build(self):
        
        self.build_system.cc = self.compiler
        self.build_system.cxx = 'g++ -std=c++17'
        rocm_include = f'-I/opt/rocm-{self.rocm_version_module}/include'
        self.build_system.cflags = [rocm_include]
        if version.parse(self.rocm_version_module) >= version.parse("7.1.0"):
            source = f"aws-ofi-nccl"
            fullpath = os.path.join(self.aws.stagedir, source)
            self.prebuild_cmds = [
                f'cp -r {fullpath} {self.stagedir}',
                f'cd {source}',
                f'module load amd/{self.aws.rocm_version_module}',
                f'module load rocm/{self.aws.rocm_version_module}'
            ]
            cflags=[f'-I/opt/rocm-{self.aws.rocm_version_module}/include']
            self.build_system.max_concurrency = 4
            self.postbuild_cmds = ['make install','cd ../lib','ln -s librccl-net.so libnccl-net.so']
            self.build_system.config_opts = [f"--with-libfabric=/opt/cray/libfabric/1.22.0 --with-rocm=/opt/rocm-{self.rocm_version_module} --prefix={self.stagedir}"]
        else:
            source = f"aws-ofi-rccl"
            fullpath = os.path.join(self.aws.stagedir, source)
            self.prebuild_cmds = [
                f'cp -r {fullpath} {self.stagedir}',
                f'cd {source}',
                f'module load amd/{self.aws.rocm_version_module}',
                f'module load rocm/{self.aws.rocm_version_module}'
            ]
            cflags=[f'-I/opt/rocm-{self.aws.rocm_version_module}/include']
            self.build_system.max_concurrency = 4
            self.postbuild_cmds = ['make install','cd ../lib','ln -s librccl-net.so libnccl-net.so']
            self.build_system.config_opts = [f"--with-libfabric=/opt/cray/libfabric/1.22.0 --prefix={self.stagedir} --with-rccl=/opt/rocm-{self.rocm_version_module}  --with-hip=/opt/rocm-{self.rocm_version_module} --disable-tests"]


class BaseSetupPyTorchDdpPipAmd(rfm.RunOnlyRegressionTest,VersionsHandlerPlugin):
    modules = [ 'cray-python']
    valid_prog_environs = ['*']
    local = True
    prerun_cmds = []
    executable = "pip"

    @run_after('setup')
    def configure_venv_and_install_pytorch(self):
        self.executable = f""" bash -exc '
            unset CUDA_VISIBLE_DEVICES;  #HACK: ROCR & CUDA devs cannot be both set
            {self.executable}
        ' """
        self.prerun_cmds.extend([
            f'python -m venv pyenv-{self.rocm_version_pytorch}',
            f'. pyenv-{self.rocm_version_pytorch}/bin/activate',
            f'pip install --upgrade pip',
            'pip install python-hostlist numpy', f'module load rocm/{self.rocm_version_module}']

        )
        self.executable = (
            f"pip install torch=={self.torch_version} torchvision "
            f"--index-url https://download.pytorch.org/whl/rocm{self.rocm_version_pytorch} "
        )
    @sanity_function
    def assert_pip_install_succeeded(self):
        return sn.assert_eq(self.job.exitcode,0)


# Warning: this is an HACK ! It intercepts CLI args to create dynamically tests
TARGET_ROCM_VER = GetCliVar('rocm_version_pytorch', '7.2')
TARGET_MOD_VER  = GetCliVar('rocm_version_module', '7.2.0')
latest_torch_versions = GetLatestPytorch(TARGET_ROCM_VER, count=3)
print(
    f"Detected rocm module version: {TARGET_MOD_VER} \n"
    f"Detected pytorch rocm version: {TARGET_ROCM_VER}"
)
print(f"Pytorch version detected {latest_torch_versions}")

SetupPyTorchDdpPipAmd = make_test(
    f'SetupPyTorchDdpPipAmd_rocm{TARGET_ROCM_VER.replace(".", "_")}',
    (BaseSetupPyTorchDdpPipAmd,),
    {
        'torch_version': parameter(latest_torch_versions),
    }
)

@rfm.simple_test
class PyTorchDdpPipAmd(PyTorchAmdTestBase,VersionsHandlerPlugin):
    descr = 'Check DDP training throughput (cnn_distr.py) on AMD MI250x'
    valid_prog_environs = ['*']
    maintainers = ['ml-team']
    num_nodes = parameter([1,2])
    venv = fixture(SetupPyTorchDdpPipAmd, scope = 'environment')
    aws = fixture(SetupAwsNccl, scope = 'environment')
    env_vars = {
        'NCCL_SOCKET_IFNAME': 'hsn0',
        'NCCL_DEBUG': 'Info',
        'NCCL_NET_GDR_LEVEL': 3,
        'NCCL_CROSS_NIC': 1,
        'NCCL_DMABUF_ENABLE': 1
     }
    time_limit = '0d0h15m0s'
    @run_after('setup')
    def configure_training_runtime(self):
        self.prerun_cmds = [ f'. {self.venv.stagedir}/pyenv-{self.rocm_version_pytorch}/bin/activate',
                    f'module load rocm/{self.rocm_version_module} amd/{self.rocm_version_module} libfabric/',
                    f'export LD_LIBRARY_PATH={self.aws.stagedir}/lib:$LD_LIBRARY_PATH',
                    ]


    
class SetupPyTorchCollective(rfm.RunOnlyRegressionTest,VersionsHandlerPlugin):
    valid_prog_environs = ['*']
    local = True
    pytorch_test = parameter([
        "allreduce-stats.py"
    ])
    executable = "curl"
    @run_after('setup')
    def configure_benchmark_download(self):
        self.executable_opts = ["-LO",f"https://raw.githubusercontent.com/IBM/pytorch-communication-benchmarks/refs/heads/main/{self.pytorch_test}"]
    @sanity_function
    def assert_download_succeeded(self):
        return sn.assert_eq(self.job.exitcode,0)
        
@rfm.simple_test
class PyTorchRCCLAllReduceVenv(rfm.RunOnlyRegressionTest,VersionsHandlerPlugin):
    descr = 'All-reduce PyTorch communication benchmark (RCCL, venv)'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    maintainers = ['ml-team']
    num_cpus_per_task = 56
    num_gpus_per_node=8
    num_tasks_per_node = 1
    time_limit = '0d0h15m0s'
    test = fixture(SetupPyTorchCollective, scope = 'environment')
    #num_nodes = variable(int, value=2)
    num_nodes = parameter([1,2])
    venv = fixture(SetupPyTorchDdpPipAmd, scope = 'environment')
    aws = fixture(SetupAwsNccl, scope = 'environment')
    env_vars = {
        'NCCL_SOCKET_IFNAME': 'hsn0',
        'NCCL_DEBUG': 'Info',
        'NCCL_NET_GDR_LEVEL': 3,
        'NCCL_CROSS_NIC': 1,
        'NCCL_DMABUF_ENABLE': 1
     }
    reference = {
        '*': {'bandwidth_avg': (91.04, -0.05, None, 'GB/s')}
    }

    @run_after('setup')
    def configure_allreduce_runtime(self):
        curr_part = self.current_partition
        self.num_gpus_per_node = curr_part.select_devices('gpu')[0].num_devices
        self.num_tasks = self.num_nodes
        self.job.options = [f'--gpus-per-task={self.num_gpus_per_node}']
        self.env_vars['OMP_NUM_THREADS'] = "1"
        headnode_cmd = (
            'masternode=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)'
        )
        self.prerun_cmds = [headnode_cmd, f'cp {self.test.stagedir}/{self.test.pytorch_test} .',
                            f'. {self.venv.stagedir}/pyenv-{self.venv.rocm_version_pytorch}/bin/activate',f'module load amd/{self.rocm_version_module}',
                            f'module load rocm/{self.rocm_version_module}', 'module load libfabric/',
                            f'export LD_LIBRARY_PATH={self.aws.stagedir}/lib:$LD_LIBRARY_PATH',
                            ] 
        self.executable = 'torchrun'
        self.executable_opts = [
            f'--nproc_per_node={self.num_gpus_per_node} ',
            f'--nnodes={self.num_nodes} ',
            '--rdzv_endpoint ${masternode} ',
            f'--rdzv_backend c10d ',
            f'{self.test.pytorch_test}', '--iterations 500 -s 2000' ]
    @sanity_function
    def assert_collective_output_present(self):
        # Use backslashes to escape the parentheses
        return sn.assert_found(r'size\(MB\)', self.stderr)
    @performance_function('GB/s')
    def bandwidth_avg(self):
        # Regex breakdown:
        # ^\s*(\d+\.\d+) matches the size (500.0) at start of line
        # \s+(?P<avg>\d+\.\d+) captures the second column as 'avg'
        return sn.extractsingle(
            r'^\s*\d+\.\d+\s+(?P<avg>\d+\.\d+)',
            self.stderr, 'avg', float
        )

    @performance_function('GB/s')
    def bandwidth_max(self):
        # Captures the third column
        return sn.extractsingle(
            r'^\s*\d+\.\d+\s+\d+\.\d+\s+(?P<max>\d+\.\d+)',
            self.stderr, 'max', float
        )

