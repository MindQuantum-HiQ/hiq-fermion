import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'


from hiqfermion.ansatzes import UCCSD
from hiqfermion.drivers import MolecularData,PyscfCalculator
from hiqfermion.utils import normal_ordered,get_fermion_operator
from hiqfermion.transforms import Transform
from hiqfermion.utils import uccsd_trotter_engine, uccsd_singlet_evolution
from hiqfermion.optimizers import minimize
from projectq.ops import X, All, Measure
import time


filename='LiH_VQE'
geometry = [['Li',[0,0,0]],
            ['H',[0,0,1]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0
molecule = MolecularData(geometry, basis, multiplicity, charge, filename=filename)

pymolecule = PyscfCalculator(molecule, posthf=['CCSD'])
pymolecule.excute()
print("LiH, ccsd:{},\tn_qubits:{},\tn_electrons:{}".format(molecule.ccsd_energy,molecule.n_qubits,molecule.n_electrons))

uccsd = UCCSD()
initial_amplitudes = uccsd.singlet_get_packed_amplitudes(molecule.ccsd_single_amps, molecule.ccsd_double_amps, molecule.n_qubits, molecule.n_electrons)
initial_amplitudes=np.array(initial_amplitudes)
print(initial_amplitudes)
print("zero:{},less than 1e-8:{},total:{}".format(sum(initial_amplitudes == 0), sum(abs(initial_amplitudes) < 1e-8), len(initial_amplitudes)))

hamiltonian =get_fermion_operator(molecule.get_molecular_hamiltonian())
qubit_hamiltonian = Transform(hamiltonian).jordan_wigner()
qubit_hamiltonian.compress()
print("terms of qubit_hamiltonian: {}".format(len(qubit_hamiltonian.terms)))

def energy_objective(packed_amplitudes,qubit_hamiltonian,n_qubits,n_electrons):
        compiler_engine=uccsd_trotter_engine()
        wavefunction = compiler_engine.allocate_qureg(n_qubits)
        for i in range(n_electrons):
            X | wavefunction[i]
        evolution_operator = uccsd_singlet_evolution(packed_amplitudes,n_qubits,n_electrons)
        evolution_operator | wavefunction
        compiler_engine.flush()
        energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
        All(Measure) | wavefunction
        compiler_engine.flush()
        return energy

def task(begin,end,packed_amplitudes,index_of_will_update,qubit_hamiltonian,n_qubits,n_electrons,epsilon=1e-8):
    t0=time.time()
    #print("\rbegin: {:3}; end:{:3}; start at {}".format(begin,end,time.ctime()),end='')
    out = []
    shift=np.zeros(packed_amplitudes.size)
    for i in range(begin,end):
        eps=0 if i==0 else epsilon
        index=0 if i==0 else i-1
        shift[index_of_will_update[index]]=eps
        out.append(energy_objective(packed_amplitudes+shift,qubit_hamiltonian,n_qubits,n_electrons))
        shift[index_of_will_update[index]]=0
    #print("\rbegin: {:3}; end:{:3}; finished at {};time used:{:6}".format(begin,end,time.ctime(),time.time()-t0),end='')
    return out

def post_processing(result,packed_amplitudes,index_of_will_update,n_qubits,n_electrons,pr,epsilon=1e-8):
    print("\nTime: {}; energy: {};".format(time.ctime(),result[0][0]))
    tmp=[result[0][0]]*len(packed_amplitudes)
    res=[i for j in result for i in j][1:]
    for i,j in enumerate(index_of_will_update):
        tmp[j]=res[i]
    if len(pr.update_iter)==0 or pr.update_iter[-1]==1:
        pr.add_record('f',result[0][0])
        pr.add_record('update_iter',0)
    return (result[0][0],(np.array(tmp)-result[0][0])/epsilon)


os.environ["OMP_NUM_THREADS"]='8'
parallel_method='multiprocessing'
record_level=2
record_file='LiH_test.data'
dump_step=1
epsilon=1e-6
# index_of_will_update=np.where(np.abs(initial_amplitudes)!=0)[0]
index_of_will_update=range(len(initial_amplitudes))
n_threads=len(index_of_will_update)+1

opti = minimize(parallel_method,record_level,record_file,dump_step)
opti.set_optimizer_para(method='L-BFGS-B', jac=True, options={'eps': epsilon})
opti.set_parallel_task(range(len(index_of_will_update) + 1), task, post_processing, n_threads)
opti.set_parallel_fix_parameters((index_of_will_update, qubit_hamiltonian,molecule.n_qubits, molecule.n_electrons, epsilon), (index_of_will_update, molecule.n_qubits, molecule.n_electrons,opti.recorder,epsilon))

t0=time.time()
result=opti.optimize(initial_amplitudes)
print("time used: {}".format(time.time()-t0))

print(result)

'''
The parameters you are using now:
 geometry:
 [['Li', [0, 0, 0]], ['H', [0, 0, 1]]]
basis:
 sto-3g
multiplicity:
 1
charge:
 0
description:
 None
filename:
 LiH_VQE
data_directory:
 None
converged SCF energy = -7.76736213574856
E(CCSD) = -7.784454825910825  E_corr = -0.01709269016226163
LiH, ccsd:-7.784454825910825,	n_qubits:12,	n_electrons:4
[ 5.98279865e-04  3.04606455e-02 -1.31950747e-18  7.94882041e-17
  2.61484705e-18  1.34571964e-16 -3.89657157e-04 -2.44649592e-04
 -1.68347176e-03 -1.12848122e-02 -8.66757597e-04 -1.83204581e-02
 -8.66757597e-04 -1.83204581e-02 -4.03694647e-04 -4.86501513e-02
 -2.62610598e-04  2.26283949e-19 -3.72219842e-21 -8.59172952e-19
  1.04969995e-18 -8.08912666e-04  2.12251773e-03 -1.75025973e-19
  2.37027414e-19  1.04736473e-18  4.73095690e-18  7.33637825e-04
  2.18382700e-02 -1.49680617e-03  5.51074933e-20  1.21164609e-19
  1.23794456e-19 -3.57237034e-19  7.54038857e-20  1.45722427e-18
 -1.54314932e-19 -4.82347585e-18 -1.49680617e-03 -5.71079930e-19
  2.16831001e-18  9.08779229e-19  2.29150535e-17 -1.47758961e-03]
zero:0,less than 1e-8:24,total:44
terms of qubit_hamiltonian: 631

Time: Tue Oct 29 17:10:13 2019; energy: -7.721524524775518;

Time: Tue Oct 29 17:10:17 2019; energy: -5.165478565499056;

Time: Tue Oct 29 17:10:21 2019; energy: -7.422439221953849;

Time: Tue Oct 29 17:10:24 2019; energy: -7.694818023664362;

Time: Tue Oct 29 17:10:28 2019; energy: -7.719863672365412;

Time: Tue Oct 29 17:10:31 2019; energy: -7.721743205960788;

Time: Tue Oct 29 17:10:35 2019; energy: -7.72234176265327;

Time: Tue Oct 29 17:10:39 2019; energy: -7.724702030218147;

Time: Tue Oct 29 17:10:42 2019; energy: -7.733596615967139;

Time: Tue Oct 29 17:10:46 2019; energy: -7.767037232759975;

Time: Tue Oct 29 17:10:50 2019; energy: -7.773439298064876;

Time: Tue Oct 29 17:10:54 2019; energy: -7.7820787845589825;

Time: Tue Oct 29 17:10:57 2019; energy: -7.782334469928358;

Time: Tue Oct 29 17:11:01 2019; energy: -7.7822125710872685;

Time: Tue Oct 29 17:11:05 2019; energy: -7.782818549982386;

Time: Tue Oct 29 17:11:08 2019; energy: -7.7832110249883035;

Time: Tue Oct 29 17:11:12 2019; energy: -7.783461407830267;

Time: Tue Oct 29 17:11:16 2019; energy: -7.783554460853254;

Time: Tue Oct 29 17:11:19 2019; energy: -7.783770598709207;

Time: Tue Oct 29 17:11:23 2019; energy: -7.784070685057524;

Time: Tue Oct 29 17:11:27 2019; energy: -7.784160877646489;

Time: Tue Oct 29 17:11:31 2019; energy: -7.784284722718677;

Time: Tue Oct 29 17:11:34 2019; energy: -7.7843413781533135;

Time: Tue Oct 29 17:11:38 2019; energy: -7.784406743637927;

Time: Tue Oct 29 17:11:42 2019; energy: -7.784417503514997;

Time: Tue Oct 29 17:11:45 2019; energy: -7.78444286105025;

Time: Tue Oct 29 17:11:49 2019; energy: -7.784445395780586;

Time: Tue Oct 29 17:11:53 2019; energy: -7.784449475845694;

Time: Tue Oct 29 17:11:57 2019; energy: -7.784450029869219;

Time: Tue Oct 29 17:12:00 2019; energy: -7.784450037376642;
time used: 108.81164073944092
      fun: -7.784450037376642
 hess_inv: <44x44 LbfgsInvHessProduct with dtype=float64>
      jac: array([-4.50004478e-05, -1.80334858e-04,  1.66428649e-04, -3.73025166e-04,
       -7.93116683e-05, -1.74183334e-04,  4.32249792e-05,  4.26743085e-05,
       -2.16128448e-04, -5.79140291e-04,  1.72800441e-04,  3.89670518e-05,
       -6.86224411e-05,  2.27373675e-05, -7.95385091e-04,  4.78265427e-04,
       -1.38760115e-05,  4.09072864e-03,  2.79260171e-04,  3.07399439e-04,
       -1.88381755e-04,  1.86704874e-04,  3.15176774e-03,  5.64041258e-04,
        6.10178574e-06,  3.39865913e-04, -7.28412886e-05, -3.08030490e-03,
       -1.64876113e-04,  4.68602934e-06, -4.49684734e-05,  1.34916966e-04,
        8.44533332e-04,  4.65387728e-04, -3.08668646e-05,  2.00828687e-04,
       -4.92629937e-04, -1.11231735e-03,  1.42916790e-05,  2.52077470e-04,
        3.15932436e-03, -1.90180227e-03, -1.02693676e-03,  8.78550566e-05])
  message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
     nfev: 30
      nit: 22
   status: 0
  success: True
        x: array([-4.58532153e-04, -2.99344853e-02,  1.19884257e-05, -6.49231171e-04,
       -2.61050136e-06, -3.94218179e-04,  5.01151238e-04, -7.01180544e-04,
        1.67790922e-03,  1.11038374e-02,  8.70455398e-04,  1.82281409e-02,
        8.65164548e-04,  1.82035742e-02,  3.82442834e-04,  4.84306041e-02,
        5.36529517e-04,  1.90098919e-04,  5.15774858e-05,  1.38384037e-05,
        2.60642404e-06,  1.61063544e-03, -4.12603450e-03,  6.37418875e-05,
        9.85467599e-05,  2.29213461e-05,  4.47513158e-05, -1.57290434e-03,
       -4.33713348e-02,  2.98409231e-03, -2.86679915e-06,  5.88120990e-06,
        3.72178578e-05,  1.14826174e-05,  2.25945385e-07,  7.84888234e-05,
       -1.66106187e-05, -4.49931241e-04,  2.98091937e-03,  1.45241233e-05,
        1.24683542e-04, -2.01121936e-05, -3.76455943e-04,  2.94796260e-03])
'''
