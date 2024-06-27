from fair.allocation import general_yankee_swap
from fair.metrics import utilitarian_welfare
import qsurvey

SPARSE = False

survey_file = "resources/random_survey.csv"
mapping_file = "resources/survey_column_mapping.csv"

mp = qsurvey.QMapper(mapping_file)
qs = qsurvey.QSurvey(survey_file)
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
schedule = mp.schedule(course_map, features)
students = qs.students(course_map, all_courses, features, schedule, SPARSE)

X = general_yankee_swap(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
