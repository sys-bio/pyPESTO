import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
from pypesto.petab import PetabImporter
from pypesto.objective.roadrunner.petab_importer_roadrunner import \
    PetabImporterRR
import petab
import pandas as pd
import tellurium as te
import matplotlib.pyplot as plt
import csv

petab_yaml = './egfr_modules/egfr.yaml'

petab_problem = petab.Problem.from_yaml(petab_yaml)
# importer = PetabImporterRR(petab_problem)

importer = PetabImporter.from_yaml(petab_yaml)
problem = importer.create_problem()
# optimizer = optimize.ScipyOptimizer()
optimizer = optimize.ScipyDifferentialEvolutionOptimizer(options={'maxiter': 100, 'popsize': 100})

# engine = pypesto.engine.SingleCoreEngine()
# optimizer = pypesto.optimize.ScipyOptimizer()

engine = pypesto.engine.MultiProcessEngine()
# optimizer = optimize.FidesOptimizer()

###########  RoadRunner  ########################################
# # importer = PetabImporterRR.from_yaml(petab_yaml)
# petab_problem = petab.Problem.from_yaml(petab_yaml)
# importer = PetabImporterRR(petab_problem)
# problem = importer.create_problem()
# engine = pypesto.engine.SingleCoreEngine()
# optimizer = pypesto.optimize.ScipyOptimizer()
#################################################################
n_starts = 100
history_options = pypesto.HistoryOptions(trace_record=True)

filename = "egfr_opt.h5"

result = optimize.minimize(
    problem=problem,
    # optimizer=optimizer_diff,
    optimizer=optimizer,
    engine=engine,
    history_options=history_options,
    n_starts=n_starts
)


# results = result.optimize_result.summary()
print(result.optimize_result[0].x)

param_dict = {}
df = pd.read_csv('./egfr_modules/parameters_b.tsv', sep='\t')
for i, each in enumerate(df.parameterId):
    param_dict[each] = 10**result.optimize_result[0].x[i]

print(param_dict)

with open('fitted_params.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in param_dict.items():
        writer.writerow([key, value])
quit()
with open('fitted_params.csv') as csv_file:
    reader = csv.reader(csv_file)
    mydict = dict(reader)

print(mydict)
quit()
model_name = './egfr_modules/EGFR_sequential_fit_egfr.ant'
new_model = ''
with open(model_name, 'r') as model:
    lines = model.readlines()
    for i, line in enumerate(lines):
        print(i, line)
        line_split = line[:-1].strip().split()
        print(i, line_split)
        if line_split and line_split[0] in param_dict:
            line_split[2] = str(param_dict[line_split[0]])
        line = ' '.join(line_split) + '\n'
        new_model += line

print(new_model)

with open(model_name, 'w') as cur_mod:
    cur_mod.write(new_model)

r = te.loada(model_name)
r.exportToSBML(model_name[:-4] + '.xml')

measurements = []
time_points = []
df = pd.read_csv('./egfr_modules/measurement_data_a.tsv', sep='\t')
for i, each in enumerate(df.measurement):
    measurements.append(each)
for i, each in enumerate(df.time):
    time_points.append(each)

print(measurements)
print(time_points)
sim = r.simulate(0, 720, 7201, ['time', 'aRtot'])
plt.scatter(time_points, measurements)
plt.plot(sim['time'], sim['aRtot'])
# plt.show()

# results = results.split('\n')
# for each in results:
#     print(each)

# result = profile.parameter_profile(
#     problem=problem,
#     result=result,
#     optimizer=optimizer,
#     engine=engine,
#     profile_index=[0, 1],
#     filename=filename
# )

# import pypesto.visualize as visualize
# from pypesto.visualize.model_fit import visualize_optimized_model_fit
# import matplotlib.pyplot as plt
#
# ref = visualize.create_references(
#     x=petab_problem.x_nominal_scaled, fval=problem.objective(petab_problem.x_nominal_scaled)
# )
# visualize.waterfall(results=result, reference=ref)
# plt.show()
# visualize.parameters(results=result, reference=ref)
# plt.show()
# visualize.profiles(results=result, reference=ref)
# plt.show()
# visualize_optimized_model_fit(petab_problem=petab_problem, result=result, pypesto_problem=result.problem)
