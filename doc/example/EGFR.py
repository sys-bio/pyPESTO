
import pypesto.optimize
import petab
import pypesto.objective.roadrunner as pypesto_rr
from pprint import pprint
import pypesto.optimize as optimize
from IPython.display import Markdown, display

model_output_dir = './egfr_new'
petab_yaml = './egfr_new/egfr.yaml'

# model_output_dir = './egfr'
# petab_yaml = './egfr/egfr.yaml'

# petab_yaml1 = './egfr/egfr1.yaml'
# petab_yaml2 = './egfr/egfr2.yaml'

# petab_problem1 = petab.Problem.from_yaml(petab_yaml1)
# petab_problem2 = petab.Problem.from_yaml(petab_yaml2)
# petab_problem = petab.CompositeProblem(problems=[petab_problem1, petab_problem2])

petab_problem = petab.Problem.from_yaml(petab_yaml)
importer = pypesto_rr.PetabImporterRR(petab_problem)
problem = importer.create_problem()

pprint(problem.objective.get_config())

# optimizer = optimize.ScipyDifferentialEvolutionOptimizer(options={'maxiter': 10, 'popsize': 10})
optimizer = optimize.ScipyOptimizer()
# engine = pypesto.engine.MultiProcessEngine(n_procs=4)
engine = pypesto.engine.SingleCoreEngine()

solver_options = pypesto_rr.SolverOptions(
    relative_tolerance=1e-6,
    absolute_tolerance=1e-12,
    maximum_num_steps=10000
)

problem.objective.solver_options = solver_options

history_options = pypesto.HistoryOptions(trace_record=True)

result = optimize.minimize(
    problem=problem,
    optimizer=optimizer,
    engine=engine,
    n_starts=200,
    history_options=history_options
)

print(result.optimize_result[0].x)


