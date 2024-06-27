import numpy as np
import pandas as pd
import re

from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.constraint import CourseTimeConstraint, MutualExclusivityConstraint
from fair.simulation import RenaissanceMan
from fair.agent import LegacyStudent
from fair.item import ScheduleItem

from qsurvey import parser


DEFAULT_CAPACITY = 30


class QSurvey:

    def __init__(self, in_file):
        df = pd.read_csv(in_file)
        self.questions = ["1", "2", "3", "4"] + [f"5#1_{i}" for i in range(1, 12)]
        self.cics_courses = [col for col in df.columns if re.match("7 _\d+$", col)]
        self.compsci_courses = [col for col in df.columns if re.match("7_\d+$", col)]
        self.info_courses = [col for col in df.columns if re.match("7 _\d\.", col)]
        self.all_courses = self.cics_courses + self.compsci_courses + self.info_courses
        self.df = df[self.questions + self.all_courses]

    def students(self, course_map, all_courses, features, schedule, sparse=False):
        course, slot, weekday, section = features
        course_time_constr = CourseTimeConstraint.from_items(
            schedule, slot, weekday, sparse
        )
        course_sect_constr = MutualExclusivityConstraint.from_items(
            schedule, course, sparse
        )

        students = []
        for idx, row in self.df.iterrows():
            preferred = [
                course_map[crs]["course num"]
                for crs in all_courses
                if not np.isnan(row[crs]) and row[crs] > 1
            ]
            topics = [preferred]
            total_num_courses = row["3"]
            if np.isnan(total_num_courses):
                continue
            student = RenaissanceMan(
                topics,
                [min(len(topic), total_num_courses) for topic in topics],
                total_num_courses,
                total_num_courses,
                course,
                [course_time_constr, course_sect_constr],
                schedule,
                seed=idx,
                sparse=sparse,
            )
            legacy_student = LegacyStudent(student, student.preferred_courses, course)
            legacy_student.student.valuation.valuation = (
                legacy_student.student.valuation.compile()
            )
            students.append(legacy_student)

        return students


class QMapper:

    def __init__(self, in_file):
        self.df = pd.read_csv(in_file, sep="|")

    def desc_for_ques(self, ques):
        return self.df[self.df.question == ques].description.values[0]

    def mapping(self, questions):
        # construct course information map
        course_map = {}
        eliminated = []
        for crs in questions:
            raw_description = self.desc_for_ques(crs)
            catalog, course_num, section, description = parser.extract_course_info(
                raw_description
            )
            instructor = parser.extract_instructor_info(raw_description)
            days, start_time, end_time = parser.extract_schedule_info(raw_description)
            try:
                course_map[crs] = {
                    "catalog": catalog,
                    "course num": course_num,
                    "section": section,
                    "description": description,
                    "instructor": instructor,
                    "days": days,
                    "time range": start_time.strftime("%I:%M %p")
                    + " - "
                    + end_time.strftime("%I:%M %p"),
                }
            except AttributeError:
                continue
                eliminated.append(crs)

        return course_map

    @staticmethod
    def features(course_map):
        # construct features
        course = Course([entry["course num"] for entry in course_map.values()])
        slot = Slot.from_time_ranges(
            [entry["time range"] for entry in course_map.values()], "15T"
        )
        weekday = Weekday()
        section = Section([entry["section"] for entry in course_map.values()])
        features = [course, slot, weekday, section]

        return features

    @staticmethod
    def schedule(course_map, features):
        # construct schedule
        schedule = []
        days = Weekday().days
        for idx, (crs, map) in enumerate(course_map.items()):
            crs = str(map["course num"])
            slt = slots_for_time_range(map["time range"], features[1].times)
            sec = map["section"]
            capacity = DEFAULT_CAPACITY
            dys = tuple([day for day in days if day in map["days"]])
            schedule.append(
                ScheduleItem(
                    features, [crs, slt, dys, sec], index=idx, capacity=capacity
                )
            )

        return schedule
