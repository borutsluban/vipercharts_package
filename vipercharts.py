# -*- coding: UTF-8 -*-
__author__ = 'borut, jan, marko'

import os, math, unicodedata, string
import webbrowser
from scipy.stats import f
from numpy import mean
from subprocess import Popen
import datetime

results_folder_name = os.path.join(os.getcwd(), os.path.normpath("results/"))

class ChartInputError(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


class CurveChart:
	objectName = "Curve Chart object"
	predTypes = ["score", "rank"]

	def __init__(self, title="Not titled yet."):
		self.title = title
		self.type = ""
		self.data = []  # list of CurveAlgo objects
		self.predictionType = "-score"  # default TODO check how this propagates and RANKs work
		self.fontSize = 11
		self._calculated = False
		self._sameName = 0

	def __setattr__(self, key, value):
		print "In Setattr with key: {0} and value: {1}".format(key, value)
		if key == "data":
			print "Setting CurveChart data"
			#if "data" in self.__dict__:
				#old = self.data
				#if value != old: # TODO include with proper comparison
			self._calculated = False
		self.__dict__[key] = value

	def __str__(self):
		return self.title + "(" + self.objectName + ")"

	def draw(self):
		template = loadChartTemplate(self.type)
		chart = fillCurveTemplate(template, self)
		#folder_name = os.path.dirname(os.path.realpath(__file__))
		file_name = os.path.join(results_folder_name, os.path.normpath(timestamp() + "-" + msgTranslate[self.type].replace(" ", "_") + "-" +cleanfilename(self.title)+".html"))
		if not os.path.exists(results_folder_name):
			os.makedirs(results_folder_name)
		if os.path.exists(file_name):
			self._sameName += 1
			file_name = file_name[0:-5]+"-({0})".format(self._sameName)+".html"
		outfile = open(file_name, 'w')
		outfile.write(chart)
		outfile.close()
		print "Output file saved to:", file_name
		webbrowser.open('file://' + outfile.name, new=2)

class ColumnChart:
	objectName = "Column Chart object"

	def __init__(self, title="Not titled yet."):
		self.title = title
		self.type = ""
		self.measureNames = [] # ["measure_name1", ...]
		self.data = []  # list of ColumnAlgo objects
		self.predictionType = "-score"  # default
		self.fontSize = 11
		self._sameName = 0

	def __str__(self):
		return self.title + "(" + self.objectName + ")"

	def draw(self):
		template = loadColumnTemplate(self.type)
		chart = fillColumnTemplate(template, self)
		file_name = os.path.join(results_folder_name, os.path.normpath(timestamp() + "-" + cleanfilename(self.title)+".html"))
		if not os.path.exists(results_folder_name):
			os.makedirs(results_folder_name)
		if os.path.exists(file_name):
			self._sameName += 1
			file_name = file_name[0:-5]+"-({0})".format(self._sameName)+".html"
		outfile = open(file_name, 'w')
		outfile.write(chart)
		outfile.close()
		print "Output file saved to:", file_name
		webbrowser.open('file://' + outfile.name, new=2)

class ScatterChart:
	objectName = "Scatter Chart object"

	def __init__(self, title="Not titled yet."):
		self.title = title
		self.type = ""
		self.data = []  # list of Algo objects
		self.predictionType = "-score"  # default
		self.fontSize = 11
		self._sameName = 0

	def __str__(self):
		return self.title + "(" + self.objectName + ")"

	def draw(self):
		template = loadScatterTemplate(self.type)
		chart = fillScatterTemplate(template, self)
		file_name = os.path.join(results_folder_name, os.path.normpath(timestamp() + "-" + cleanfilename(self.title)+".html"))
		if not os.path.exists(results_folder_name):
			os.makedirs(results_folder_name)
		if os.path.exists(file_name):
			self._sameName += 1
			file_name = file_name[0:-5]+"-({0})".format(self._sameName)+".html"
		outfile = open(file_name, 'w')
		outfile.write(chart)
		outfile.close()
		print "Output file saved to:", file_name
		webbrowser.open('file://' + outfile.name, new=2)

class CompareChart:
	objectName = "Compare Chart object"

	def __init__(self, title="Not titled yet."):
		self.title = title
		self.measureName = ""
		self.type = "" # type = "2" for Wilcoxon and type = "n" for Friedman and Nemenyi post hoc test
		self.numberOfAlgorithms = None
		self.numberOfDataSets = None
		self.alpha = 0.05
		self.data = []  # list of Algo objects
		self.predictionType = "-score"  # default

	def __str__(self):
		return self.title + "(" + self.objectName + ")"

	def draw(self, toWhere = None):
		# check user input
		if toWhere not in ["pdf", "toString", "html", None]:
			raise ChartInputError("Argument should be \"pdf\" for critical difference diagram in pdf format, "
								  "or \"toString\" for Latex source code for critical difference diagram, "
								  "or \"html\" for visualization in browser.")
		if self.alpha not in [0.5, 0.4, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005]:
			raise ChartInputError("Significance level alpha can take one of the following numbers: "
								  "0.5, 0.4, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005.")
		if self.type not in ["2", "n"]:
			raise ChartInputError("Set type to \"2\" to compare 2 algorithms with Wilcoxon test "
								  "or set type to \"n\" to compare n algorithms with Friedman and Nemenyi post hoc test.")
		if toWhere == "pdf":
			if self.type == "2":
				raise ChartInputError("Type should be \"n\". Critical difference diagram in pdf format is "
									  "possible only for comparison of n algorithms "
									  "with Friedman and Nemenyi post hoc test.")
			dict = {}
			for algo in self.data:
				dict[algo.name] = algo.measureValues
			dict['names'] = dict.keys()
			data = comparenAlgorithms([dict], self.alpha)[0]
			exportComparison(toWhere, [data], str(self.alpha).replace(".", "-"))
			print "File critDiff.pdf is in directory: viperchartspackage/vipercharts/exports/"
		elif toWhere == "toString":
			if self.type == "2":
				raise ChartInputError("Type should be \"n\". Latex source code for critical difference diagram is "
									  "possible only for comparison of n algorithms "
									  "with Friedman and Nemenyi post hoc test.")
			dict = {}
			for algo in self.data:
				dict[algo.name] = algo.measureValues
			dict['names'] = dict.keys()
			data = comparenAlgorithms([dict], self.alpha)[0]
			exportComparison(toWhere, [data], str(self.alpha).replace(".", "-"))
			file_object = open("/home/marko/viperchartspackage/vipercharts/exports/critDiff.tex", "r")
			print file_object.read()
		else: # html
			template = loadCompareTemplate(self.type)
			chart = fillCompareTemplate(template, self, self.type)
			file_name = os.path.join(results_folder_name, os.path.normpath(timestamp() + "-" + cleanfilename(self.title)+".html"))
			if not os.path.exists(results_folder_name):
				os.makedirs(results_folder_name)
			outfile = open(file_name, 'w')
			outfile.write(chart)
			outfile.close()
			print "Output file saved to:", file_name
			webbrowser.open('file://' + outfile.name, new=2)

class CurveAlgo:
	objectName = "CurveAlgo object"

	def __setattr__(self, key, value):
		print "In CurveAlgo Setattr with key: {0} and value: {1}".format(key, value)
		if key == "actualList" or key == "predictedList":
			print "Changing CurveAlgo input data (actual or predicted)"
			#if key in self.__dict__:
				#old = self.data
				#if value != old: # TODO include with proper comparison
			self._dataChange = True
		self.__dict__[key] = value

	def __init__(self, name="Not named yet", actualList=[], predictedList=[], probabList=[],
				 folds=True, thresholdAverage=False, microAverage=False, multiClass=False, samples=None):
		self.name = str(name).strip()
		#self._dataChange = False
		self.actualList = actualList
		self.predictedList = predictedList
		self.probabList = probabList
		self.data = [] # list of CurveData objects		#
		self.multiClass = multiClass
		self.folds = folds
		self.thresholdAverage = thresholdAverage
		self.microAverage = microAverage
		self.samples = samples
		# if not self.folds and not self.thresholdAverage and not self.microAverage:
		# 	raise ChartInputError("Set \"folds\" to True (plot all curves) or " \
		# 						  + "\"microAverage\" to True (treat all results as if they came from a single iteration) or " \
		# 						  + "\"thresholdAverage\" to True (traverse over threshold and average the curves positions at them).")

		if len(actualList) != len(predictedList):
			raise ChartInputError("Lists must have the same length.")
		if probabList != [] and len(probabList) != len(actualList):
			raise ChartInputError("Lists must have the same length.")

		if multiClass:
			if type(probabList[0][0]) == type([]):  # class distribution probability estimates
				curveDataList = []
				if probabList == []:
					raise ChartInputError("Multi-class data needs a list of probabilities.")
				for i in range(len(actualList)):
					aset = set(actualList[i])
					num_of_classes = len(aset)
					for a in aset:
						if type(a) != type(1):
							raise ChartInputError("Actual list should be a list of integers.")
					for p in predictedList[i]:
						if type(p) != type(1):
							raise ChartInputError("Predicted list should be a list of integers.")
					for prob in probabList[i]:
						if len(prob) != num_of_classes:
							raise ChartInputError("Probability list should have a score or probability for each class.")
						for pr in prob:
							if type(pr) != type(1) and type(pr) != type(1.0):
								raise ChartInputError("Probability list should be a list of doubles.")

					# change multi-class data to 1-vs-all data
					a_in = actualList[i]
					p_in = predictedList[i]
					prob_in = probabList[i]

					for class_num in range(1, num_of_classes + 1, 1):
						a_out = []
						prob_out = []
						for i in range(len(a_in)):
							# prob_in[i] = [prob for class 1, ..., prob for class num_of_classes]
							prob_out.append(prob_in[i][class_num - 1])
							if a_in[i] == class_num:
								a_out.append(1)
							else:
								a_out.append(0)
						curveDataList.append(
							CurveData(self.name + " curve of class " + str(class_num - 1), a_out, prob_out,
									  multiClass=True))
				self.data = curveDataList  # list of CurveData objects

			else:  # not class distribution probability estimates
				curveDataList = []
				if probabList is []:
					raise ChartInputError("Multi-class data needs a list of probabilities.")
				for i in range(len(actualList)):
					aset = set(actualList[i])
					for a in aset:
						if type(a) != type(1):
							raise ChartInputError("Actual list should be a list of integers.")
					for p in predictedList[i]:
						if type(p) != type(1):
							raise ChartInputError("Predicted list should be a list of integers.")
					for prob in probabList[i]:
						if prob > 1 or prob < 0:
							raise ChartInputError("Probability list should be a list of probabilities.")

					# change multi-class data to 1-vs-all data
					a_in = actualList[i]
					p_in = predictedList[i]
					prob_in = probabList[i]
					num_of_classes = len(set(a_in))
					for class_num in range(1, num_of_classes + 1, 1):
						a_out = []
						prob_out = []
						for i in range(len(a_in)):
							if a_in[i] == class_num:
								a_out.append(1)
								if p_in[i] == class_num:  # we predict class_num with prob_in
									prob_out.append(prob_in[i])
								else:  # we predict class_num with approx (1 - prob_in)/num_of_classes
									prob_out.append((1 - prob_in[i]) / (num_of_classes - 1))
							else:
								a_out.append(0)
								if p_in[
									i] != class_num:  # we predict class_num with approx (1 - prob_in)/num_of_classes
									prob_out.append((1 - prob_in[i]) / (num_of_classes - 1))
								else:  # a_in != class_num and p_in = class_num -> we predict class_num with prob_in
									prob_out.append(prob_in[i])
						curveDataList.append(
							CurveData(self.name + " curve of class " + str(class_num - 1), a_out, prob_out,
									  multiClass=True))
				self.data = curveDataList  # list of CurveData objects

		else:  # not multi-class data
			if len(actualList) > 0:
				# if folds/repetitions data
				if type(actualList[0]) == type([]):
					curveDataList = []
					if not self.thresholdAverage and not self.microAverage:
						print "WARNING: No averaging selected. Set " \
		 						  + "\"microAverage\" to True (treat all results as if they came from a single iteration) or " \
		 						  + "\"thresholdAverage\" to True (traverse over threshold and average the curves positions at them). " \
								  + "To NOT show folds/iterations set \"folds\" to False (plot only averaged curves)."
					for i in range(len(actualList)):
						aset = set(actualList[i])
						if (aset != set([0, 1]) and aset != set([0]) and aset != set([1])):
							raise ChartInputError(
								"Actual list should be a list of 0's and 1's. If data is multi-class, set multiClass=True.")
						for p in predictedList[i]:
							if type(p) != type(1) and type(p) != type(1.0):
								raise ChartInputError("Predicted list should be a list of integers or doubles.")
						if len(actualList[i]) != len(predictedList[i]):
							raise ChartInputError("Lists must have the same length at fold/iteration {0}.".format(i))
						if len(probabList) == len(actualList):
							curveDataList.append(CurveData(name + " " + str(i), actualList[i], predictedList[i], probabList[i]))
						else:
							curveDataList.append(CurveData(name + " " + str(i), actualList[i], predictedList[i]))
					self.data = curveDataList  # list of CurveData objects
				# else, just one series of predictions
				else:
					aset = set(actualList)
					if (aset != set([0, 1]) and aset != set([0]) and aset != set([1])):
						raise ChartInputError(
							"Actual list should be a list of 0's and 1's. If data is multi-class, set multiClass=True.")
					for p in predictedList:
						if type(p) != type(1) and type(p) != type(1.0):
							raise ChartInputError("Predicted list should be a list of integers or doubles.")
					self.data = [CurveData(self.name, actualList, predictedList)]
			else:
				raise ChartInputError("List contains no data.")

		if self.microAverage:
			mergedActual = []
			mergedPredicted = []
			for cd in self.data:
				mergedActual.append(cd.actual)
				mergedPredicted.append(cd.predicted)
			flattenedActual = [val for sublist in mergedActual for val in sublist]
			flattenedPredicted = [val for sublist in mergedPredicted for val in sublist]
			self.data.append(
				CurveData(self.name + " (micro-average)", flattenedActual, flattenedPredicted, microAverage=True))

	def __str__(self):
		minl = min(len(self.actualList[0]), 10)
		shortened = (minl < len(self.actualList[0]))
		actualString = ""
		predictedString = ""
		for l in self.actualList:
			actualString += str(l[:minl]) + (" ..., " if shortened else "")
		for l in self.predictedList:
			predictedString += str(l[:minl]) + (" ..., " if shortened else "")
		return self.name + " (" + self.objectName + ")" \
			   + "\n\t actualList: " + actualString \
			   + "\n\t predictedList: " + predictedString

class ColumnAlgo:
	objectName = "ColumnAlgo object"

	def __init__(self, name="Not named yet", measureValues=[]):
		self.name = str(name).strip()
		self.measureValues = measureValues
		# TODO preveri, ce je dolzina measureValues == measureNames
		for p in measureValues:
			if type(p) != type(1) and type(p) != type(1.0):
				raise ChartInputError("Measure values should be a list of doubles.")

	def __str__(self):
		return self.name + " (" + self.objectName + ")" \
			   + "\n\t measureValues: " + str(self.measureValues)

class ScatterAlgo:
	objectName = "ScatterAlgo object"

	def __init__(self, name="Not named yet", measureValues=[]):
		self.name = str(name).strip()
		self.measureValues = measureValues
		# TODO preveri, ce je dolzina measureValues == measureNames
		for p in measureValues:
			if type(p) != type(1) and type(p) != type(1.0):
				raise ChartInputError("Measure values should be a pair of doubles.")

	def __str__(self):
		return self.name + " (" + self.objectName + ")" \
			   + "\n\t measureValues: " + str(self.measureValues)

class CompareAlgo:
	objectName = "CompareAlgo object"

	def __init__(self, name="Not named yet", measureValues=[]):
		self.name = str(name).strip()
		self.measureValues = measureValues
		for p in measureValues:
			if type(p) != type(1) and type(p) != type(1.0):
				raise ChartInputError("Measure values should be a list of doubles.")

	def __str__(self):
		return self.name + " (" + self.objectName + ")" \
			   + "\n\t measureValues: " + str(self.measureValues)


class CurveData:
	objectName = "Curve Data object"

	def __init__(self, name="Not named yet", actual=[], predicted=[], probab=[],
				 microAverage=False, thresholdAverage=False, multiClass=False):
		self.name = str(name).strip()
		self.actual = actual
		self.predicted = predicted
		self.probab = probab
		self.performance = {}
		self.microAverage = microAverage
		self.thresholdAverage = thresholdAverage
		self.multiClass = multiClass

	def __str__(self):
		minl = min(len(self.actual), 10)
		shortened = (minl < len(self.actual))
		return self.name + " (" + self.objectName + ")" \
			   + "\n\tactual: " + str(self.actual[:minl]) + (" ... " if shortened else "") \
			   + "\n\tpredicted: " + str(self.predicted[:minl]) + (" ... " if shortened else "")


def cleanfilename(s):
	validChars = "-_.() {0}{1}".format(string.ascii_letters, string.digits)
	toascii = unicodedata.normalize("NFKD",unicode(s,'utf-8')).encode("ASCII","ignore")
	return ''.join([c for c in toascii if c in validChars]).replace(" ","_")

def timestamp():
	return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def prepareCurveData(curveChart, subtype):
	nPoints = 4
	performance = curveChart.data
	kenmax = 0.5
	ratemax = 0.5
	list_of_pi0 = []

	# performance is a list [alg1, alg2, ...]
	for algo in performance:
		print algo.name, algo.data
		for curve in [c for c in algo.data if not c.thresholdAverage]:  # TODO parallel? maybe setup takes longer?
			print curve
			n = len(curve.actual)  # curve['actual']
			if type(curve.actual[0]) != int:  # curve['actual']
				curve.actual = map(int, curve.actual)  # curve['actual']
			negs = curve.actual.count(0)  # curve['actual']
			poss = curve.actual.count(1)  # curve['actual']
			if poss == 0 or negs == 0:
				# print "Class Error, zero poss or zero negs, only one class or other type error."
				return []
			try:
				ranks = curve['rank']
			except:
				ranks = range(n + 1)[1:]  # ranks from 1
			parallel = []
			for i in range(n):
				parallel.append([curve.actual[i], float(curve.predicted[i])])  # curve['actual'] #curve['predicted']
			_oldrate = 0
			_oldloss = 0
			AUC = 0
			AUPR = 0
			tp = 0;
			fp = 0;
			tp_old = 0;
			fp_old = 0;
			n1 = 0;
			concordant_pairs = 0;
			discordant_pairs = 0;

			if (subtype == '-score'):
				ROCseries = [[0, 0, '-Inf']];
				PRseries = [[0, 1, '-Inf']];
				LIFTseries = [[0, 0, '-Inf']]
				ROChull = [[0, 0, '-Inf']];
				COSTseries = [[0, 0, '-Inf']];
				RATEseries = [];
				KENseries = [[0, 0, '-Inf']];
				KENup = [[0, 1]];
				KENdown = [[0, 0]]

				ranked = sorted(parallel, key=lambda pair: pair[1], reverse=True)
				pi0 = poss * 1.0 / n
				pi1 = 1 - pi0
				if not curve.microAverage:
					list_of_pi0.append(pi0)

				k = 0
				while k < len(ranked):
					addedconc = 0;
					addeddisc = 0;
					threshold = ranked[k][1];
					group = [x[0] for x in ranked if x[1] >= threshold]
					tp = group.count(1)
					fp = group.count(0)

					ties = len(group) - k
					n1 += ties * (ties - 1) / 2
					concordant_pairs += tp_old * (fp - fp_old)
					discordant_pairs += fp_old * (tp - tp_old)

					ROCpoint = [fp * 1.0 / negs, tp * 1.0 / poss, threshold]
					ROCseries.append(ROCpoint)
					AUC += (ROCpoint[1] + ROCseries[-2][1]) * (ROCpoint[0] - ROCseries[-2][0]) * 0.5
					PRseries.append([tp * 1.0 / poss, tp * 1.0 / (tp + fp), threshold])
					AUPR += (PRseries[-1][1] + PRseries[-2][1]) * (PRseries[-1][0] - PRseries[-2][0]) * 0.5
					LIFTseries.append([len(group) * 1.0 / n, tp * 1.0 / poss, threshold])
					# Convex hull and lower envelope
					while len(ROChull) >= 2 and (ROChull[-1][0] == ROCpoint[0] or (
									ROChull[-2][0] != ROChull[-1][0] and (ROChull[-1][1] - ROChull[-2][1]) / (
										ROChull[-1][0] - ROChull[-2][0]) <= (ROCpoint[1] - ROChull[-1][1]) / (
										ROCpoint[0] - ROChull[-1][0]))):
						ROChull.pop()
						COSTseries.pop()
					ROChull.append(ROCpoint)
					if (ROCpoint[0] != ROChull[-2][0]):
						slope = (ROCpoint[1] - ROChull[-2][1]) / (ROCpoint[0] - ROChull[-2][0])
						intercept = ROCpoint[1] - slope * ROCpoint[0]
						COSTseries.append([1 / (slope + 1), (1 - intercept) / (1 + slope), threshold])
					else:
						if len(COSTseries) == 0:
							COSTseries.append([0, 0, threshold])
						else:
							COSTseries[0][2] = threshold
					COSTend = 1 - ROCpoint[1]

					# Rate-driven curve:
					# The Rate-driven curve is a list of intervals. Each interval is a set of points
					# on the appropriate parabola. There are nPoints number of points
					RATEinterval = []

					_newrate = pi1 * ROCpoint[0] + pi0 * ROCpoint[1]
					_newloss = 2 * (_newrate * (pi0 - _newrate) + pi1 * ROCpoint[0])
					RATEinterval.append([_oldrate, _oldloss, threshold, algo.data.index(curve) + 1])

					for i in range(1, nPoints):
						alpha = i * 1.0 / nPoints
						rate = _oldrate + alpha * (_newrate - _oldrate)
						loss = 2 * (
							rate * (pi0 - rate) + pi1 * (ROCseries[-2][0] + alpha * (ROCpoint[0] - ROCseries[-2][0])))
						RATEinterval.append([rate, loss, 0])

					RATEinterval.append([_newrate, _newloss, 0])
					RATEseries.append(RATEinterval)

					# ratemax
					if _newloss > ratemax:
						ratemax = _newloss
					m = 0.5 * (pi0 + pi1 * (ROCseries[-2][0] - ROCpoint[0]) / (_newrate - _oldrate))
					if m < _newrate and m > _oldrate:
						mvalue = 2 * (
							m * (pi0 - m) + pi1 * ((_newrate - m) * ROCseries[-2][0] + (m - _oldrate) * ROCpoint[0]) / (
								_newrate - _oldrate))
						if mvalue > ratemax:
							ratemax = mvalue

					# Kendall curve:
					if _newrate <= pi0:
						KENseries.append([_newrate, 2 * pi1 * ROCpoint[0], threshold])
					else:
						if _oldrate < pi0:
							KENseries.append([pi0,
											  (2 * pi1 * ROCpoint[0] - KENseries[-1][1]) * (pi0 - KENseries[-1][0]) / (
												  _newrate - KENseries[-1][0]) + (KENseries[-1][1]), ''])
						KENseries.append([_newrate, 2 * pi0 * (1 - ROCpoint[1]), threshold])
					if KENseries[-1][1] > kenmax:
						kenmax = KENseries[-1][1]
					_oldrate = _newrate
					_oldloss = _newloss

					k += len(group) - k
					tp_old = tp
					fp_old = fp
			else:
				ROCseries = [[0, 0, 0]];
				PRseries = [[0, 1, 0]];
				LIFTseries = [[0, 0, 0]]  # x: y: rank:
				ROChull = [[0, 0, 0]];
				COSTseries = [[0, 0, 0]];
				RATEseries = [];
				KENseries = [[0, 0]];
				KENup = [[0, 1]];
				KENdown = [[0, 0]]
				ranked = sorted(parallel, key=lambda pair: pair[1])
				# print ranked
				k = 0
				while k < len(ranked):
					tp = 0;
					fp = 0;
					threshold = ranked[k][1];
					group = [x[0] for x in ranked if x[1] <= threshold]
					tp = group.count(1)
					fp = group.count(0)

					ties = len(group) - k
					n1 += ties * (ties - 1) / 2
					concordant_pairs += tp_old * (fp - fp_old)
					discordant_pairs += fp_old * (tp - tp_old)

					ROCpoint = [fp * 1.0 / negs, tp * 1.0 / poss, threshold]
					ROCseries.append([fp * 1.0 / negs, tp * 1.0 / poss, int(threshold)])
					AUC += (ROCpoint[1] + ROCseries[-2][1]) * (ROCpoint[0] - ROCseries[-2][0]) * 0.5
					PRseries.append([tp * 1.0 / poss, tp * 1.0 / (tp + fp), int(threshold)])
					AUPR += (PRseries[-1][1] + PRseries[-2][1]) * (PRseries[-1][0] - PRseries[-2][0]) * 0.5
					LIFTseries.append([len(group) * 1.0 / n, tp * 1.0 / poss, int(threshold)])

					# Convex hull and lower envelope:
					while len(ROChull) >= 2 and (ROChull[-1][0] == ROCpoint[0] or (
									ROChull[-2][0] != ROChull[-1][0] and (ROChull[-1][1] - ROChull[-2][1]) / (
										ROChull[-1][0] - ROChull[-2][0]) <= (ROCpoint[1] - ROChull[-1][1]) / (
										ROCpoint[0] - ROChull[-1][0]))):
						ROChull.pop()
						COSTseries.pop()
					ROChull.append(ROCpoint)
					if (ROCpoint[0] != ROChull[-2][0]):
						slope = (ROCpoint[1] - ROChull[-2][1]) / (ROCpoint[0] - ROChull[-2][0])
						intercept = ROCpoint[1] - slope * ROCpoint[0]
						COSTseries.append([1 / (slope + 1), (1 - intercept) / (1 + slope), threshold])
					else:
						if len(COSTseries) == 0:
							COSTseries.append([0, 0, threshold])
						else:
							COSTseries[0][2] = threshold
					COSTend = 1 - ROCpoint[1]

					# Rate-driven curve:
					# The Rate-driven curve is a list of intervals. Each interval is a set of points on the appropriate parabola. There are nPoints number of points
					RATEinterval = []
					pi0 = poss * 1.0 / n
					pi1 = 1 - pi0
					_newrate = pi1 * ROCpoint[0] + pi0 * ROCpoint[1]
					_newloss = 2 * (_newrate * (pi0 - _newrate) + pi1 * ROCpoint[0])
					RATEinterval.append([_oldrate, _oldloss, threshold, performance.index(curve) + 1])
					for i in range(1, nPoints):
						alpha = i * 1.0 / nPoints
						rate = _oldrate + alpha * (_newrate - _oldrate)
						loss = 2 * (
							rate * (pi0 - rate) + pi1 * (ROCseries[-2][0] + alpha * (ROCpoint[0] - ROCseries[-2][0])))
						RATEinterval.append([rate, loss, 0])
					RATEinterval.append([_newrate, _newloss, 0])
					RATEseries.append(RATEinterval)

					if _newloss > ratemax:
						ratemax = _newloss
					m = 0.5 * (pi0 + pi1 * (ROCseries[-2][0] - ROCpoint[0]) / (_newrate - _oldrate))
					if m < _newrate and m > _oldrate:
						mvalue = 2 * (
							m * (pi0 - m) + pi1 * ((_newrate - m) * ROCseries[-2][0] + (m - _oldrate) * ROCpoint[0]) / (
								_newrate - _oldrate))
						if mvalue > ratemax:
							ratemax = mvalue

					# Kendall curve:
					if _newrate <= pi0:
						KENseries.append([_newrate, 2 * pi1 * ROCpoint[0], threshold])
					else:
						if _oldrate < pi0:
							KENseries.append([pi0,
											  (2 * pi1 * ROCpoint[0] - KENseries[-1][1]) * (pi0 - KENseries[-1][0]) / (
												  _newrate - KENseries[-1][0]) + (KENseries[-1][1]), ''])
						KENseries.append([_newrate, 2 * pi0 * (1 - ROCpoint[1]), threshold])
					if KENseries[-1][1] > kenmax:  # check outputs

						kenmax = KENseries[-1][1]
					_oldrate = _newrate
					_oldloss = _newloss

					k += len(group) - k
					tp_old = tp
					fp_old = fp

			if COSTseries[-1][0] < 1:
				# append final point with max threshold
				COSTseries.append([1, COSTend, ranked[-1][1]])

			curve.performance['ROCpoints'] = ROCseries
			curve.performance['PRpoints'] = PRseries
			curve.performance['LIFTpoints'] = LIFTseries
			curve.performance['ROChull'] = ROChull
			curve.performance['COSTpoints'] = COSTseries
			curve.performance['RATEintervals'] = RATEseries
			curve.performance['KENpoints'] = KENseries
			curve.performance['AUC'] = AUC
			curve.performance['Gini'] = 2 * AUC - 1
			n0 = n * (n - 1) / 2
			try:
				curve.performance['KENtau'] = (concordant_pairs - discordant_pairs) / math.sqrt(
					(n0 - n1) * (n0 - (negs * (negs - 1) + poss * (poss - 1)) / 2))
			except:
				print "ZeroDivisionError when calculation KENtau"
				curve.performance['KENtau'] = None
			curve.performance['AUPR'] = AUPR
			AUCH = 0
			for i in range(1, len(ROChull)):
				AUCH += (ROChull[i][1] + ROChull[i - 1][1]) * (ROChull[i][0] - ROChull[i - 1][0]) * 0.5
			curve.performance['AUCH'] = AUCH
			curve.performance['KENmax'] = kenmax  # performance[0]['KENmax'] = kenmax
			curve.performance['RATEmax'] = ratemax  # performance[0]['RATEmax'] = ratemax



		#############################################################
		# threshold average
		#############################################################
		allCurves = len(algo.data)
		withoutTHcurve = [c for c in algo.data if c.thresholdAverage == False]
		if algo.thresholdAverage and algo._dataChange: # not to compute threshold curve multiple times
			# we make one average curve from all the curves in algo.data except average curves
			folds_algo_data = []

			thIndex = -1
			for i, curve in enumerate(algo.data):
				if curve.thresholdAverage:
					thIndex = i
				if not curve.microAverage and not curve.thresholdAverage:
					folds_algo_data.append(curve)

			num_of_curves = len(folds_algo_data)
			if thIndex >= 0:
				del algo.data[thIndex]

			# append curves to lists of curves
			ROC_curves = []
			PR_curves = []
			LIFT_curves = []
			for curve in folds_algo_data:
				dict = curve.performance
				# we skip first point [0,0,'-Inf'] and add it at the end to the average curve
				ROC_curves.append(dict['ROCpoints'][1:])
				PR_curves.append(dict['PRpoints'][1:])
				LIFT_curves.append(dict['LIFTpoints'][1:])


			# all scores = thresholds
			ROC_thresholds = []
			for roc in ROC_curves:
				ROC_thresholds.append([roc[i][2] for i in range(len(roc))])
			ROC_thresholds = sorted(list(set([item for sublist in ROC_thresholds for item in sublist])), reverse=True)

			PR_thresholds = []
			for pr in PR_curves:
				PR_thresholds.append([pr[i][2] for i in range(len(pr))])
			PR_thresholds = sorted(list(set([item for sublist in PR_thresholds for item in sublist])), reverse=True)

			LIFT_thresholds = []
			for lift in LIFT_curves:
				LIFT_thresholds.append([lift[i][2] for i in range(len(lift))])
			LIFT_thresholds = sorted(list(set([item for sublist in LIFT_thresholds for item in sublist])), reverse=True)


			# number of points in each curve
			ROC_npts = [len(ROC_curves[i]) for i in range(len(ROC_curves))]
			PR_npts = [len(PR_curves[i]) for i in range(len(PR_curves))]
			LIFT_npts = [len(LIFT_curves[i]) for i in range(len(LIFT_curves))]


			# average ROC curve
			ROC_avg = []
			for tidx in range(0, len(ROC_thresholds)):
				FPsum = 0
				TPsum = 0
				for i in range(num_of_curves):
					p = pointAtThresh(ROC_curves[i], ROC_npts[i], ROC_thresholds[tidx])
					FPsum += p[0]
					TPsum += p[1]
				ROC_avg.append([FPsum / num_of_curves, TPsum / num_of_curves, ROC_thresholds[tidx]])
			ROC_avg = [[0, 0, '-Inf']] + ROC_avg

			# from average ROC curve to ROC hull, COST, rate driven, kendall
			nPoints = 4
			kenmax = 0.5
			_oldrate = 0
			_oldloss = 0
			n = len(ROC_avg) - 1  # len(actual) == len(ROCseries) - 1
			pi0 = mean(list_of_pi0)
			pi1 = 1 - pi0

			ROChull_avg = [[0, 0, '-Inf']]
			COST_avg = [[0, 0, '-Inf']]
			RATE_avg = []
			KEN_avg = [[0, 0, '-Inf']]

			for k in range(1, len(ROC_avg)):
				ROCpoint = ROC_avg[k]
				threshold = ROCpoint[2]
				# print "threshold = {0} at k = {1}".format(threshold, k)

				# ROC hull
				while len(ROChull_avg) >= 2 and (ROChull_avg[-1][0] == ROCpoint[0] or (
								ROChull_avg[-2][0] != ROChull_avg[-1][0] and (
							ROChull_avg[-1][1] - ROChull_avg[-2][1]) / (
									ROChull_avg[-1][0] - ROChull_avg[-2][0]) <= (ROCpoint[1] - ROChull_avg[-1][1]) / (
									ROCpoint[0] - ROChull_avg[-1][0]))):
					ROChull_avg.pop()
					COST_avg.pop()
				ROChull_avg.append(ROCpoint)


				# COST curve
				if (ROCpoint[0] != ROChull_avg[-2][0]):
					slope = (ROCpoint[1] - ROChull_avg[-2][1]) / (ROCpoint[0] - ROChull_avg[-2][0])
					intercept = ROCpoint[1] - slope * ROCpoint[0]
					COST_avg.append([1 / (slope + 1), (1 - intercept) / (1 + slope), threshold])
				else:
					if len(COST_avg) == 0:
						COST_avg.append([0, 0, threshold])
					else:
						COST_avg[0][2] = threshold


				# Rate-driven curve
				RATEinterval = []
				_newrate = pi1 * ROCpoint[0] + pi0 * ROCpoint[1]
				_newloss = 2 * (_newrate * (pi0 - _newrate) + pi1 * ROCpoint[0])
				RATEinterval.append(
					[_oldrate, _oldloss, threshold, 1])  # TODO 1 namesto algo.data.index(curve) + 1, ker 1 krivulja?

				for i in range(1, nPoints):
					alpha = i * 1.0 / nPoints
					rate = _oldrate + alpha * (_newrate - _oldrate)
					loss = 2 * (
						rate * (pi0 - rate) + pi1 * (ROC_avg[k - 1][0] + alpha * (ROCpoint[0] - ROC_avg[k - 1][0])))
					RATEinterval.append([rate, loss, 0])

				RATEinterval.append([_newrate, _newloss, 0])
				RATE_avg.append(RATEinterval)


				# Kendall curve
				if _newrate <= pi0:
					KEN_avg.append([_newrate, 2 * pi1 * ROCpoint[0], threshold])
				else:
					if _oldrate < pi0:
						KEN_avg.append([pi0,
										(2 * pi1 * ROCpoint[0] - KEN_avg[-1][1]) * (pi0 - KEN_avg[-1][0]) / (
											_newrate - KEN_avg[-1][0]) + (KEN_avg[-1][1]), ''])
					KEN_avg.append([_newrate, 2 * pi0 * (1 - ROCpoint[1]), threshold])
				if KEN_avg[-1][1] > kenmax:
					kenmax = KEN_avg[-1][1]

				_oldrate = _newrate
				_oldloss = _newloss


			# average PR curve
			PR_avg = []
			for tidx in range(0, len(PR_thresholds)):
				Recall_sum = 0
				Precision_sum = 0
				for i in range(num_of_curves):
					p = pointAtThresh(PR_curves[i], PR_npts[i], PR_thresholds[tidx])
					Recall_sum += p[0]
					Precision_sum += p[1]
				PR_avg.append([Recall_sum / num_of_curves, Precision_sum / num_of_curves, PR_thresholds[tidx]])
			PR_avg = [[0, 1, '-Inf']] + PR_avg

			# average LIFT curve
			LIFT_avg = []
			for tidx in range(0, len(LIFT_thresholds)):
				RelSampleSize_sum = 0  # TODO je to okej?
				TPsum = 0
				for i in range(num_of_curves):
					p = pointAtThresh(LIFT_curves[i], LIFT_npts[i], LIFT_thresholds[tidx])
					RelSampleSize_sum += p[0]
					TPsum += p[1]
				LIFT_avg.append([RelSampleSize_sum / num_of_curves, TPsum / num_of_curves, LIFT_thresholds[tidx]])
			LIFT_avg = [[0, 0, '-Inf']] + LIFT_avg

			average_curve = CurveData(algo.name + " (threshold-average)", thresholdAverage=True)
			average_curve.performance['ROCpoints'] = ROC_avg
			average_curve.performance['ROChull'] = ROChull_avg  # ROChull_avg ali ROChull_avg2_hull
			average_curve.performance['PRpoints'] = PR_avg
			average_curve.performance['LIFTpoints'] = LIFT_avg
			average_curve.performance['COSTpoints'] = COST_avg
			average_curve.performance['RATEintervals'] = RATE_avg  # folds_algo_data[0].performance['RATEintervals']
			average_curve.performance['KENpoints'] = KEN_avg

			algo.data.append(average_curve)
		# END algo curves
		algo._dataChange = False

		# if not folds remove folds( = non average curves)
		if not algo.folds:
			nofolds_algo_data = []
			for curve in algo.data:
				if curve.microAverage or curve.thresholdAverage:
					nofolds_algo_data.append(curve)
			algo.data = nofolds_algo_data
	# END algorithms
	curveChart._calculated = True
	return performance


def pointAtThresh(ROC, npts, thresh):
	i = 0
	while i < npts - 1 and ROC[i][2] > thresh:
		i += 1
	return ROC[i]


def getColumns(row):
	columns = []
	j = 0
	while j < len(row) and row[j].strip() not in columns:
		columns.append(row[j].strip())
		j += 1
	return columns


translate = {'rocc': 'roc', 'roch': 'roc', 'prc': 'prcurve', 'lift': 'lift', 'cost': 'cost', 'kendall': 'ken',
			 'ratedriven': 'rate', 'rocs': 'roc-space', 'prs': 'pr-space'}
pointsTranslate = {'rocc': 'ROCpoints', 'roch': 'ROChull', 'prc': 'PRpoints', 'lift': 'LIFTpoints',
				   'cost': 'COSTpoints', 'kendall': 'KENpoints', 'ratedriven': 'RATEintervals', 'rocs': 'roc-space',
				   'prs': 'pr-space'}
msgTranslate = {'rocc': 'ROC', 'roch': 'ROC hull', 'prc': 'PR', 'lift': 'LIFT', 'cost': 'COST', 'kendall': 'KENDALL',
				'ratedriven': 'RATE-DRIVEN'}



'''Template manipulation'''


def loadChartTemplate(type):
	fromdir = os.path.dirname(os.path.realpath(__file__))
	frame = open(fromdir + os.path.normpath('/charts/api-frame.html'), 'r')
	ttext = ""
	line = frame.readline()
	while line.strip() != '{% block content%}{% endblock %}':
		ttext += line
		line = frame.readline()
	template = open(fromdir + os.path.normpath('/charts/api-curve-' + translate[type] + '.html'), 'r')
	ttext += "".join(template.readlines()[3:-2])  # without django template lines
	ttext += '</body>\r\n</html>'
	return ttext


def loadColumnTemplate(type):
	fromdir = os.path.dirname(os.path.realpath(__file__))
	frame = open(fromdir + os.path.normpath('/charts/api-frame.html'), 'r')
	ttext = ""
	line = frame.readline()
	while line.strip() != '{% block content%}{% endblock %}':
		ttext += line
		line = frame.readline()

	template = open(fromdir + os.path.normpath('/charts/api-column.html'), 'r') #TODO spremeni v api-column brez 2
	ttext += "".join(template.readlines()[3:-2])  # without django template lines
	ttext += '</body>\r\n</html>'
	return ttext


def loadScatterTemplate(type):  # type \in {pr-space, roc-space}
	fromdir = os.path.dirname(os.path.realpath(__file__))
	frame = open(fromdir + os.path.normpath('/charts/api-frame.html'), 'r')
	ttext = ""
	line = frame.readline()
	while line.strip() != '{% block content%}{% endblock %}':
		ttext += line
		line = frame.readline()

	template = open(fromdir + os.path.normpath('/charts/api-scatter-' + type + '.html'), 'r')
	ttext += "".join(template.readlines()[3:-2])  # without django template lines
	ttext += '</body>\r\n</html>'
	return ttext


def loadCompareTemplate(type):
	fromdir = os.path.dirname(os.path.realpath(__file__))
	frame = open(fromdir + os.path.normpath('/charts/api-frame.html'), 'r')
	ttext = ""
	line = frame.readline()
	while line.strip() != '{% block content%}{% endblock %}':
		ttext += line
		line = frame.readline()

	template = open(fromdir + os.path.normpath('/charts/api-compare-' + type + '.html'), 'r')
	ttext += "".join(template.readlines()[3:-2])  # without django template lines
	ttext += '</body>\r\n</html>'
	return ttext


def fillCurveTemplate(temp, curveChart):
	# Calculate curve data
	if not curveChart._calculated:
		performance = prepareCurveData(curveChart, '-score')  # performance is a list [alg1, alg2, ...]
	else:
		performance = curveChart.data

	# fill template
	temp = temp.replace("{{ showLegend }}", 'true')
	temp = temp.replace("{{ title }}",
						curveChart.title.strip())  # if ('title' in pdata.keys() and pdata['title'].strip() != "") else '')
	temp = temp.replace("{{ chart.subtype }}", 'rank' if (
		curveChart.predictionType.strip() != "-score") else '-score')  # if ('predtype' in pdata.keys() and pdata['predtype'].strip() != "rank") else '-score')
	temp = temp.replace("{{ fontSize }}", str(
		curveChart.fontSize))  # if ('fontsize' in pdata.keys() and int(pdata['fontsize']) in range(5,50)) else 11)
	if curveChart.type == 'ratedriven':
		algo1 = performance[0]
		curve1 = algo1.data[0]
		temp = temp.replace("{{ RATEmax }}", str(curve1.performance["RATEmax"]))  # curveChart.performance[0]["RATEmax"]
		temp = temp.replace("{{ RATEintervals }}", prepareRateDrivenPoints(performance))
	else:
		if curveChart.type == 'kendall':
			algo1 = performance[0]
			curve1 = algo1.data[0]
			temp = temp.replace('{{ data.0.KENmax }}', str(curve1.performance[
															   'KENmax']))  # pepointsTranslate = {'rocc':'ROCpoints', 'roch':'ROChull', 'prc':'PRpoints', 'lift':'LIFTpoints', 'cost':'COSTpoints', 'kendall':'KENpoints','ratedriven':'RATEintervals', 'rocs':'roc-space', 'prs':'pr-space'}rformance[0].['KENmax']
		# TODO correct fbeta handling
		if curveChart.type == 'prc':  # TODO possibility to set other beta values
			temp = temp.replace('{{ fbeta }}', str(1))

		# curve points
		temp = temp.replace("{{ curvePoints }}", prepareCurveDataPoints(performance, curveChart.type))

	return temp


def fillColumnTemplate(temp, columnChart):
	temp = temp.replace("{{ showLegend }}", 'true')
	temp = temp.replace("{{ title }}",
						columnChart.title.strip())  # if ('title' in pdata.keys() and pdata['title'].strip() != "") else '')
	temp = temp.replace("{{ fontSize }}", str(
		columnChart.fontSize))  # if ('fontsize' in pdata.keys() and int(pdata['fontsize']) in range(5,50)) else 11)

	pstring = ""
	for algo in columnChart.data:
		pstring += '\'' + algo.name + '\', ' # '[alg1', 'alg2', 'alg3]',
	temp = temp.replace("{{ algos }}", pstring[1:-3])

	pstring = ""
	for i in range(len(columnChart.measureNames)): # i for columns
		# columnChart.measureNames[i]
		datastring = ""
		for j in range(len(columnChart.data)): # j for number of algorithms
			datastring += '''{ y: ''' + str(columnChart.data[j].measureValues[i]) + ''', index: ''' + str(j) + ''' }, '''
		pstring += '''
			{
				name: \'''' + columnChart.measureNames[i] + '''\',
				data: [''' + datastring + ''']
			},'''
	temp = temp.replace("{{ series }}", pstring)
	return temp


def fillScatterTemplate(temp, scatterChart):
	temp = temp.replace("{{ showLegend }}", 'true')
	temp = temp.replace("{{ title }}",
						scatterChart.title.strip())  # if ('title' in pdata.keys() and pdata['title'].strip() != "") else '')
	temp = temp.replace("{{ fontSize }}", str(
		scatterChart.fontSize))  # if ('fontsize' in pdata.keys() and int(pdata['fontsize']) in range(5,50)) else 11)

	if scatterChart.type == "pr-space":
		# points for corresponding F-isolines
		pstring = ""
		for algo in scatterChart.data:
			pstring += '''[''' + str(algo.measureValues[0]) + ''', ''' + str(algo.measureValues[1]) + '''], \n'''
		temp = temp.replace("{{ points }}", pstring)

		# points
		pstring = ""
		for algo in scatterChart.data:
			pstring += '''
			{
				type: 'scatter',
				name: \'''' + algo.name + '''\',
				data: [{ x: ''' + str(algo.measureValues[0]) + ''', y: ''' + str(algo.measureValues[1]) + '''}]
			},'''
		temp = temp.replace("{{ series }}", pstring)
	else:
		# points
		pstring = ""
		for algo in scatterChart.data:
			pstring += '''
			{
				type: 'scatter',
				name: \'''' + algo.name + '''\',
				data: [[''' + str(algo.measureValues[0]) + ''', ''' + str(algo.measureValues[1]) + '''],]
			},'''
		temp = temp.replace("{{ series }}", pstring)
	return temp


def fillCompareTemplate(temp, compareChart, type):
	dict = {}
	for algo in compareChart.data:
		dict[algo.name] = algo.measureValues
	dict['names'] = dict.keys()
	if type == "2":
		performance = compare2Algorithms([dict], compareChart.alpha)[0]
		print performance
		temp = temp.replace("{{ numberOfDataSets }}", str(compareChart.numberOfDataSets))
		temp = temp.replace("{{ critValue }}", str(performance['critValue']))
		temp = temp.replace("{{ signCritValue }}", str(performance['signCritValue']))
		temp = temp.replace("{{ measureName }}", str(compareChart.measureName))
		# tbody
		temp = temp.replace("{{ Tvalue }}", str(performance['Tvalue']))
		if performance['Tratio'] > performance['critValue']: # item.Tratio.0 > item.critvalue TODO: je to okej?
			temp = temp.replace("{{ verdictWilcoxon }}", str(compareChart.data[1].name) + " is better") # counted for the second algorithm
		else:
			temp = temp.replace("{{ verdictWilcoxon }}", "undecided")
		# {{ signTestResult }} replace with "2 : 10 (with 2 ties)" (wins are counted for the second algorithm)
		temp = temp.replace("{{ signTestResult }}", str(performance['signLoss']) + " : " + str(performance['signWins'])
							+ " (with " + str(performance['signTies']) + " ties)")
		# {{ verdictSignTest }} item.signWins > item.signCritValue
		if performance['signWins'] > performance['critValue']: # TODO: je to okej? Lahko popravim >= in dodam polovico ties v wins
			temp = temp.replace("{{ verdictSignTest }}", str(compareChart.data[1].name) + " is better")
		else:
			temp = temp.replace("{{ verdictSignTest }}", "undecided")
		temp = temp.replace("{{ Tmax }}", str(performance['Tmax']))

		temp = temp.replace("{{ alg0 }}", compareChart.data[0].name)
		temp = temp.replace("{{ data0 }}", str(max(performance['Tratio'])))

	else:
		performance = comparenAlgorithms([dict], compareChart.alpha)[0]
		temp = temp.replace("{{ title }}",
							compareChart.title.strip())  # if ('title' in pdata.keys() and pdata['title'].strip() != "") else '')
		temp = temp.replace("{{ numberOfAlgorithms }}", str(compareChart.numberOfAlgorithms))
		temp = temp.replace("{{ numberOfDataSets }}", str(compareChart.numberOfDataSets))
		temp = temp.replace("{{ alpha }}", str(compareChart.alpha))
		temp = temp.replace("{{ friedmanF }}", str(round(performance['friedmanF'], 4)))
		temp = temp.replace("{{ friedmanP }}", str(round(performance['friedmanP'], 4)))
		if performance['max'] - performance['min'] >= performance['critDist']: #TODO: je to okej?
			temp = temp.replace("{{ significance }}","Differences in classifier performance are significant")
		else:
			temp = temp.replace("{{ significance }}","There is no significant difference in algorithm performance")
		temp = temp.replace("{{ xmin }}", str(performance['min']))
		temp = temp.replace("{{ xmax }}", str(performance['max']))
		temp = temp.replace("{{ critDistRounded }}", str(round(performance['critDist'],3)))
		temp = temp.replace("{{ critDist }}", str(performance['critDist']))

		# series1
		pstring = ""
		averageRanks = performance['averageRanks']
		for rank in averageRanks:
			pstring += '''
			{
				name: \'''' + rank[0] + '''\',
				data: [[ '''+ str(rank[1]) + ''', 0]],
				type: 'scatter'
			},
			'''
		temp = temp.replace("{{ series1 }}", pstring)

		# series2
		pstring = ""
		k = 0 # the number of groups
		groups = performance['groups']
		for group in groups:
			pstring += '''
			{
				marker: { enabled: false },
				name: \'''' + str(group['index']) + '''\',
				showInLegend: false,
				data: [
					[''' + str(group['start']) + ''', -1 * ''' + str((0.4 + k*0.2)) + '''],
					[''' + str(group['end']) + ''', -1 * ''' + str((0.4 + k*0.2)) + '''],
				]
			},
			'''
			k += 1
		temp = temp.replace("{{ series2 }}", pstring)

		# chartSize or height of the chart depends on the number of groups k
		chartSize = 290 + k*20 # chartSize is 290 + 20 for each additional group
		ymin = -0.009 * chartSize + 2.1 # ymin(chartSize) = -0.009 * chartSize + 2.1
		temp = temp.replace("{{ chartSize }}", str(chartSize))
		temp = temp.replace("{{ ymin }}", str(ymin))

		# tooltip1
		pstring = ""
		for rank in averageRanks:
			pstring += '''
				if(this.series.name == \'''' + rank[0] + '''\'){
					return '<b>''' + rank[0] + '''</b><br/>Average rank: '''+ str(round(rank[1], 3)) + '''\';
				}
			'''
		temp = temp.replace("{{ tooltip1 }}", pstring)

		# tooltip2
		pstring = ""
		for group in groups:
			ppstring = ""
			for classifier in group['classifiers']:
				ppstring += '''
					s = s + \'''' + classifier + ''', ';

				'''
			pstring += '''
				if(this.series.name == \'''' + str(group['index']) + '''\'){
					var s = '<b> Group ''' + str(group['index']) + ''':</b><br/>';

					''' + ppstring + '''

					return s.slice(0,-2);
				}
			'''
		temp = temp.replace("{{ tooltip2 }}", pstring)
	return temp



def prepareFisoPoints(data, type):
	pstring = ""
	for item in data:
		for point in item[pointsTranslate[type]]:
			pstring += '''
				[''' + str(point[0]) + ', ' + str(point[1]) + '],'
	return pstring

def prepareCurveDataPoints(performance, type):
	pstring = ""
	for algo in performance:  # performance is a list [alg1, alg2, ...]
		for item in algo.data:  # item is a curve
			pstring += '''
			{
				name: \'''' + item.name + '''\',
				data: ['''
			for point in item.performance[pointsTranslate[type]]:
				pstring += '''
					{ x:''' + str(point[0]) + ', y:' + str(point[1]) + ', tr:\'' + str(
					point[2]) + '\' },'  # dodal str(point[2])
			pstring += '''
				]
			},'''
	return pstring


def prepareRateDrivenPoints(performance):  # TODO popravi barve, da ne ponovi
	pstring = ""
	for algo in performance:  # performance is a list [alg1, alg2, ...]
		for item in algo.data:  # item is a curve
			for interval in item.performance["RATEintervals"]:
				pstring += '''
				{
					'''
				if interval[0][0]:
					pstring += "showInLegend: false,"
				pstring += '''
					color: Highcharts.getOptions().colors[''' + str(interval[0][3]) + ''' - 1],
					name: \'''' + item.name + '''\',
					marker: {
						enabled: true,
						symbol: Highcharts.getOptions().symbols[''' + str(interval[0][3]) + ''']
						},
					data: ['''
				for point in interval:
					if point[2]:
						pstring += '''
						{ x: ''' + str(point[0]) + ", y: " + str(point[1]) + ", tr: " + str(
							interval[0][2]) + ", showtooltip: true, serieslength: " + str(
							len(item.performance["RATEintervals"])) + " },"
					else:
						pstring += '''
						{
							x: ''' + str(point[0]) + ''',
							y: ''' + str(point[1]) + ''',
							showtooltip: false,
							marker: {
								enabled: false,
								//Workaround: Due to highcharts bug hover cannot be disabled, so the image is a 1x1 transparent .png
								symbol: 'url(/charts/spot.png)'
								},
						},'''
				pstring += '''
					]
				},'''
	return pstring


#############################################################
# support for statistical significance tests
#############################################################

WILCOXON_VALUES = {
	0.5: [0, 0, 0, 0, 5, 7.5, 10.5, 14, 18, 22.5, 27.5, 33, 39, 45.5, 52.5, 60, 68, 76.5, 85.5, 95, 105, 115.5, 126.5,
		  138,
		  150, 162.5, 175.5, 189, 203, 217.5, 232.5, 248, 264, 280.5, 297.5, 315, 333, 351.5, 370.5, 390, 410, 430.5,
		  451.5,
		  473, 495, 517.5, 540.5, 564, 588, 612.5, 637.5],
	0.4: [0, 0, 0, 0, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 48, 55, 63, 71, 80, 89, 98, 108, 119, 130, 141, 153, 165,
		  178,
		  192, 206, 220, 235, 250, 266, 282, 299, 317, 335, 353, 372, 391, 411, 431, 452, 473, 495, 517, 540, 563, 587,
		  611],
	0.3: [0, 0, 0, 0, 3, 5, 8, 11, 14, 18, 22, 27, 32, 38, 44, 51, 58, 65, 73, 82, 91, 100, 110, 120, 131, 143, 155,
		  167,
		  180, 193, 207, 221, 236, 251, 266, 283, 299, 316, 334, 352, 371, 390, 409, 429, 450, 471, 492, 514, 536, 559,
		  583],
	0.2: [0, 0, 0, 0, 3, 4, 6, 9, 12, 15, 19, 23, 28, 33, 39, 45, 51, 58, 66, 74, 83, 91, 100, 110, 120, 131, 142, 154,
		  166,
		  178, 191, 205, 219, 233, 248, 263, 279, 295, 312, 329, 347, 365, 384, 403, 422, 442, 463, 484, 505, 527, 550],
	0.1: [0, 0, 0, 0, 1, 3, 4, 6, 9, 11, 15, 18, 22, 27, 32, 37, 43, 49, 56, 63, 70, 78, 87, 95, 105, 114, 125, 135,
		  146,
		  158, 170, 182, 195, 208, 222, 236, 251, 266, 282, 298, 314, 331, 349, 366, 385, 403, 423, 442, 463, 483, 504],
	0.05: [0, 0, 0, 0, 0, 1, 3, 4, 6, 9, 11, 14, 18, 22, 26, 31, 36, 42, 48, 54, 61, 68, 76, 84, 92, 101, 111, 120, 131,
		   141, 152, 164, 176, 188, 201, 214, 228, 242, 257, 272, 287, 303, 320, 337, 354, 372, 390, 408, 428, 447,
		   467],
	0.025: [0, 0, 0, 0, 0, 0, 1, 3, 4, 6, 9, 11, 14, 18, 22, 26, 30, 35, 41, 47, 53, 59, 67, 74, 82, 90, 99, 108, 117,
			127,
			138, 148, 160, 171, 183, 196, 209, 222, 236, 250, 265, 280, 295, 311, 328, 344, 362, 379, 397, 416, 435],
	0.01: [0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 6, 8, 10, 13, 16, 20, 24, 28, 33, 38, 44, 50, 56, 63, 70, 77, 85, 94, 102, 111,
		   121, 131, 141, 152, 163, 175, 187, 199, 212, 225, 239, 253, 267, 282, 297, 313, 329, 346, 363, 381, 398],
	0.005: [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 6, 8, 10, 13, 16, 20, 24, 28, 33, 38, 44, 49, 55, 62, 69, 76, 84, 92, 101,
			110,
			119, 129, 139, 149, 160, 172, 184, 196, 208, 221, 235, 248, 263, 277, 292, 308, 324, 340, 357, 374]}

SIGN_TEST_VALUES = {
	0.5: [-1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
		  19,
		  19, 20, 20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28],
	0.4: [-1, -1, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18,
		  19,
		  19, 20, 20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29],
	0.3: [-1, -1, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18, 18, 19,
		  19,
		  20, 20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30],
	0.2: [-1, -1, -1, 4, 5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18, 18, 19, 20,
		  20,
		  21, 21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 31],
	0.1: [-1, -1, -1, -1, 5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 16, 17, 18, 18, 19, 19, 20,
		  20,
		  21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 28, 29, 30, 30, 31, 31, 32],
	0.05: [-1, -1, -1, -1, -1, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 18, 19, 20, 20,
		   21,
		   21, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 29, 30, 31, 31, 32, 32, 33],
	0.025: [-1, -1, -1, -1, -1, -1, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 18, 19, 20, 20, 21,
			21,
			22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34],
	0.01: [-1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 20, 21, 22,
		   22,
		   23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 30, 30, 31, 31, 32, 33, 33, 34, 34, 35],
	0.005: [-1, -1, -1, -1, -1, -1, -1, -1, 9, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22,
			22,
			23, 24, 24, 25, 25, 26, 27, 27, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 35, 35, 36], }

STUDENTIZED_RANGE_VALUES = {
	0.5: [0, 0, 0.6744898, 1.122735, 1.398884, 1.595857, 1.747722, 1.870617, 1.973422, 2.061526, 2.138433, 2.206549,
		  2.267588, 2.322817, 2.373197, 2.419472, 2.46223, 2.501942, 2.538994, 2.573704, 2.606335, 2.637112, 2.666224,
		  2.693834, 2.720081, 2.745088, 2.768961, 2.791794, 2.813669, 2.83466, 2.854832],
	0.4: [0, 0, 0.8416212, 1.291403, 1.562905, 1.755341, 1.90329, 2.022844, 2.122776, 2.208381, 2.283091, 2.349254,
		  2.408543, 2.462191, 2.511133, 2.556091, 2.597636, 2.636227, 2.672238, 2.705977, 2.7377, 2.767624, 2.795933,
		  2.822784, 2.848315, 2.872641, 2.895868, 2.918085, 2.939373, 2.959803, 2.979439],
	0.3: [0, 0, 1.036433, 1.481099, 1.745705, 1.932444, 2.075754, 2.191464, 2.288149, 2.370962, 2.443237, 2.507249,
		  2.564619, 2.616539, 2.663913, 2.70744, 2.74767, 2.785047, 2.819932, 2.852621, 2.883364, 2.912368, 2.939812,
		  2.965847, 2.990605, 3.0142, 3.036732, 3.058288, 3.078945, 3.098773, 3.117831],
	0.2: [0, 0, 1.281552, 1.713694, 1.968414, 2.147707, 2.285184, 2.396161, 2.488898, 2.568347, 2.637706, 2.699156,
		  2.754247, 2.804122, 2.849644, 2.891482, 2.930163, 2.966112, 2.999674, 3.031132, 3.060724, 3.08865, 3.11508,
		  3.140159, 3.164014, 3.186753, 3.208471, 3.229253, 3.249173, 3.268296, 3.286681],
	0.1: [0, 0, 1.644854, 2.052293, 2.291341, 2.459516, 2.588521, 2.692732, 2.779884, 2.854606, 2.919889, 2.977768,
		  3.029694, 3.076733, 3.119693, 3.159199, 3.195743, 3.229723, 3.261461, 3.291224, 3.319233, 3.345676, 3.370712,
		  3.394477, 3.417089, 3.438651, 3.459253, 3.478971, 3.497878, 3.516033, 3.533492],
	0.05: [0, 0, 1.959964, 2.343701, 2.569032, 2.727774, 2.849705, 2.94832, 3.030879, 3.10173, 3.163684, 3.218654,
		   3.268004,
		   3.312739, 3.353618, 3.39123, 3.426041, 3.458425, 3.488685, 3.517073, 3.543799, 3.56904, 3.592946, 3.615646,
		   3.637252, 3.657861, 3.677556, 3.696413, 3.714498, 3.731869, 3.748578],
	0.025: [0, 0, 2.241403, 2.603757, 2.817124, 2.967746, 3.083626, 3.177472, 3.256126, 3.323692, 3.382822, 3.435327,
			3.482496, 3.525279, 3.564396, 3.600406, 3.633749, 3.66478, 3.693789, 3.721013, 3.746652, 3.770875, 3.793823,
			3.815621, 3.836374, 3.856173, 3.875101, 3.893227, 3.910615, 3.92732, 3.943392],
	0.01: [0, 0, 2.575829, 2.913494, 3.11325, 3.254686, 3.36374, 3.452213, 3.526471, 3.590339, 3.646292, 3.696021,
		   3.740733,
		   3.781318, 3.818451, 3.852654, 3.884343, 3.91385, 3.941446, 3.967357, 3.99177, 4.014842, 4.03671, 4.057487,
		   4.077275, 4.096161, 4.11422, 4.131519, 4.148118, 4.164069, 4.17942],
	0.005: [0, 0, 2.807034, 3.128407, 3.319221, 3.45463, 3.559207, 3.644155, 3.715528, 3.776968, 3.830833, 3.878738,
			3.921835, 3.960974, 3.9968, 4.029814, 4.060412, 4.088913, 4.115578, 4.140622, 4.164225, 4.186538, 4.207691,
			4.227794, 4.246945, 4.265225, 4.282709, 4.299461, 4.315538, 4.33099, 4.345863]}

STANDARD_NORMAL_VALUES = {
	0.5: [0.67448975, 1.15034938, 1.382994127, 1.534120544, 1.644853627, 1.731664396, 1.802743091, 1.862731867,
		  1.914505825,
		  1.959963985, 2.000423569, 2.036834132, 2.069901831, 2.100165493, 2.128045234, 2.153874694, 2.177923069,
		  2.200410581, 2.221519588, 2.241402728, 2.260188991, 2.277988333, 2.294895209, 2.310991338, 2.326347874,
		  2.341027138, 2.355084009, 2.368567059, 2.38151947, 2.3939798, 2.405982615, 2.417559016, 2.428737087,
		  2.439542264,
		  2.449997661, 2.460124338, 2.469941537, 2.479466885, 2.488716566, 2.497705474, 2.506447346, 2.514954878,
		  2.523239824, 2.531313091, 2.539184814, 2.546864427, 2.554360729, 2.561681935, 2.568835728],
	0.4: [0.841621234, 1.281551566, 1.501085946, 1.644853627, 1.750686071, 1.833914636, 1.902216496, 1.959963985,
		  2.009874772, 2.053748911, 2.092837799, 2.128045234, 2.160044423, 2.189349756, 2.216362779, 2.241402728,
		  2.26472742, 2.286547951, 2.307039259, 2.326347874, 2.344597707, 2.361894447, 2.378328933, 2.3939798,
		  2.408915546,
		  2.423196195, 2.436874627, 2.449997661, 2.462606936, 2.474739649, 2.486429155, 2.497705474, 2.508595724,
		  2.519124473, 2.529314052, 2.539184814, 2.548755358, 2.558042727, 2.567062573, 2.575829304, 2.584356209,
		  2.592655576, 2.600738783, 2.608616387, 2.616298204, 2.623793369, 2.631110406, 2.638257273, 2.645241416],
	0.3: [1.036433389, 1.439531471, 1.644853627, 1.780464342, 1.880793608, 1.959963985, 2.025099553, 2.080278453,
		  2.128045234, 2.170090378, 2.207592001, 2.241402728, 2.272158761, 2.300346956, 2.326347874, 2.350464423,
		  2.372941497, 2.3939798, 2.413745804, 2.432379059, 2.449997661, 2.466702404, 2.482579981, 2.497705474,
		  2.512144328,
		  2.525953917, 2.539184814, 2.551881811, 2.564084769, 2.575829304, 2.587147367, 2.598067731, 2.608616387,
		  2.618816898, 2.628690684, 2.638257273, 2.647534521, 2.656538788, 2.665285106, 2.673787315, 2.682058187,
		  2.690109527, 2.697952275, 2.705596583, 2.713051888, 2.720326982, 2.727430061, 2.734368787, 2.741150323],
	0.2: [1.281551566, 1.644853627, 1.833914636, 1.959963985, 2.053748911, 2.128045234, 2.189349756, 2.241402728,
		  2.286547951, 2.326347874, 2.361894447, 2.3939798, 2.423196195, 2.449997661, 2.474739649, 2.497705474,
		  2.519124473,
		  2.539184814, 2.558042727, 2.575829304, 2.592655576, 2.608616387, 2.623793369, 2.638257273, 2.652069808,
		  2.665285106, 2.677950909, 2.690109527, 2.701798628, 2.713051888, 2.723899532, 2.734368787, 2.744484261,
		  2.754268271, 2.763741112, 2.772921295, 2.781825748, 2.790469991, 2.798868286, 2.807033768, 2.814978562,
		  2.822713881, 2.830250116, 2.837596913, 2.844763242, 2.851757461, 2.858587364, 2.865260239, 2.871782901],
	0.1: [1.644853627, 1.959963985, 2.128045234, 2.241402728, 2.326347874, 2.3939798, 2.449997661, 2.497705474,
		  2.539184814,
		  2.575829304, 2.608616387, 2.638257273, 2.665285106, 2.690109527, 2.713051888, 2.734368787, 2.754268271,
		  2.772921295, 2.790469991, 2.807033768, 2.822713881, 2.837596913, 2.851757461, 2.865260239, 2.878161739,
		  2.890511561, 2.902353479, 2.913726318, 2.924664667, 2.935199469, 2.945358513, 2.955166847, 2.964647126,
		  2.973819901, 2.982703875, 2.991316115, 2.999672235, 3.007786556, 3.015672247, 3.02334144, 3.030805337,
		  3.038074305, 3.045157952, 3.052065202, 3.058804356, 3.065383152, 3.071808808, 3.078088073, 3.084227265],
	0.05: [1.959963985, 2.241402728, 2.3939798, 2.497705474, 2.575829304, 2.638257273, 2.690109527, 2.734368787,
		   2.772921295, 2.807033768, 2.837596913, 2.865260239, 2.890511561, 2.913726318, 2.935199469, 2.955166847,
		   2.973819901, 2.991316115, 3.007786556, 3.02334144, 3.038074305, 3.052065202, 3.065383152, 3.078088073,
		   3.090232306, 3.101861834, 3.113017263, 3.12373463, 3.134046055, 3.143980287, 3.153563159, 3.162817966,
		   3.171765783, 3.180425743, 3.188815259, 3.196950229, 3.204845205, 3.212513537, 3.219967505, 3.227218426,
		   3.234276754, 3.241152166, 3.247853632, 3.254389487, 3.260767488, 3.266994864, 3.273078364, 3.279024298,
		   3.284838574],
	0.025: [2.241402728, 2.497705474, 2.638257273, 2.734368787, 2.807033768, 2.865260239, 2.913726318, 2.955166847,
			2.991316115, 3.02334144, 3.052065202, 3.078088073, 3.101861834, 3.12373463, 3.143980287, 3.162817966,
			3.180425743, 3.196950229, 3.212513537, 3.227218426, 3.241152166, 3.254389487, 3.266994864, 3.279024298,
			3.290526731, 3.301545185, 3.312117666, 3.32227792, 3.332056035, 3.341478956, 3.350570901, 3.359353718,
			3.36784718, 3.376069242, 3.384036252, 3.391763141, 3.399263576, 3.406550105, 3.413634266, 3.420526701,
			3.427237241, 3.433774987, 3.44014838, 3.446365263, 3.452432937, 3.458358209, 3.464147434, 3.469806555,
			3.475341139],
	0.01: [2.575829304, 2.807033768, 2.935199469, 3.02334144, 3.090232306, 3.143980287, 3.188815259, 3.227218426,
		   3.260767488, 3.290526731, 3.317247362, 3.341478956, 3.363635456, 3.384036252, 3.402932835, 3.420526701,
		   3.436981724, 3.452432937, 3.466992901, 3.480756404, 3.493804004, 3.506204727, 3.518018159, 3.529296089,
		   3.540083799, 3.550421113, 3.560343232, 3.569881421, 3.579063572, 3.587914672, 3.596457188, 3.604711395,
		   3.612695651, 3.62042663, 3.627919521, 3.635188198, 3.642245368, 3.649102696, 3.655770917, 3.662259931,
		   3.668578887, 3.674736256, 3.6807399, 3.686597123, 3.692314726, 3.697899051, 3.703356022, 3.708691178,
		   3.71390971],
	0.005: [2.807033768, 3.02334144, 3.143980287, 3.227218426, 3.290526731, 3.341478956, 3.384036252, 3.420526701,
			3.452432937, 3.480756404, 3.506204727, 3.529296089, 3.550421113, 3.569881421, 3.587914672, 3.604711395,
			3.62042663, 3.635188198, 3.649102696, 3.662259931, 3.674736256, 3.686597123, 3.697899051, 3.708691178,
			3.719016485, 3.728912783, 3.738413501, 3.747548342, 3.756343809, 3.76482365, 3.773009224, 3.780919811,
			3.788572872, 3.795984269, 3.803168454, 3.810138631, 3.816906894, 3.823484349, 3.829881219, 3.836106931,
			3.842170201, 3.848079098, 3.853841114, 3.859463207, 3.864951862, 3.870313123, 3.875552638, 3.880675688,
			3.885687223]}

TOL = 1e-10
def compare2Algorithms(data, alpha):
	performance = data
	algorithms = [item for item in performance[0].keys() if item not in ['measure', 'names', 'Tvalue']]
	# performance[0]['names'] = names
	for measure in performance:
		# diff = [x - y for x,y in zip(measure[names[0]], measure[names[1]])]
		sorteddiff = sorted([[float(x) - float(y), 0] for x, y in zip(measure[algorithms[0]], measure[algorithms[1]])],
							key=lambda x: abs(x[0]))
		n = len(sorteddiff)
		tiestart = 0

		signTies = 0
		signWins = 0
		for i in range(1, n):
			if abs(abs(sorteddiff[i][0]) - abs(sorteddiff[tiestart][0])) < TOL:
				# found a tie.
				pass
			else:
				# no tie. Assign ranks to previous tied group.
				for j in range(tiestart, i):
					sorteddiff[j][1] = 1 + 1.0 * (i - 1 + tiestart) / 2
				tiestart = i
		# assign ranks to last tied group:
		for j in range(tiestart, n):
			sorteddiff[j][1] = 1 + 1.0 * (n - 1 + tiestart) / 2
		Rp = 0;
		i = 0;
		zrank = sorteddiff[0][1]
		while (i + 1 < n and sorteddiff[i + 1][0] == 0):
			# for each 2 zero values, add one zrank to both R+ and R-.
			Rp += zrank
			i += 2
		signTies = i;
		# go to first nonzero element:
		if i < n and sorteddiff[i][0] == 0:
			i += 1
		Rm = Rp
		for j in range(i, n):
			if sorteddiff[j][0] > 0:
				Rp += sorteddiff[j][1]
				signWins += 1;
			else:
				Rm += sorteddiff[j][1]
		Tmax = n * (n + 1) / 4.0
		measure['Tmax'] = Tmax

		if Rp < Rm:
			measure['Tvalue'] = Rp
			measure['Tratio'] = [Rp - Tmax, 0]
		else:
			measure['Tvalue'] = Rm
			measure['Tratio'] = [0, Tmax - Rm]

		measure['signTies'] = signTies
		measure['signWins'] = signWins
		measure['signLoss'] = n - signTies - signWins
		measure['signCritValue'] = SIGN_TEST_VALUES[alpha][n - 1]
		# default pvalue is 0.05
		measure['critValue'] = WILCOXON_VALUES[alpha][n - 1]
	return performance


def comparenAlgorithms(data, alpha):
	performance = data
	classifiers = performance[0]['names']
	nClassifiers = len(classifiers)
	for measure in performance:
		rankdict = dict([(classifier, 0) for classifier in classifiers])
		nDataSets = len(measure[classifiers[0]])
		for i in range(nDataSets):
			addRanks([float(measure[classifier][i]) for classifier in classifiers], classifiers, rankdict)
		for classifier in classifiers:
			rankdict[classifier] = rankdict[classifier] * 1.0 / nDataSets
		measure['averageRanks'] = rankdict.items()
		chisq = 12.0 * nDataSets * (
		sum([rank ** 2 for rank in rankdict.values()]) - (nClassifiers * (nClassifiers + 1) ** 2) / 4.0) / (
				nClassifiers * (nClassifiers + 1))
		friedmanF = (nDataSets - 1) * chisq / (nDataSets * (nClassifiers - 1) - chisq)
		measure['chisq'] = chisq
		measure['friedmanF'] = friedmanF
		pvalue = 1 - f.cdf(friedmanF, nClassifiers - 1, (nClassifiers - 1) * (nDataSets - 1))
		measure['friedmanP'] = pvalue
		measure['pvalue'] = alpha
		measure['min'] = min(rankdict.values())
		measure['max'] = max(rankdict.values())
		# calculate groups:
		measure['critDist'] = math.sqrt(nClassifiers * (nClassifiers + 1.0) / (6 * nDataSets)) * \
							  STUDENTIZED_RANGE_VALUES[alpha][nClassifiers]
		measure['groups'] = calculateGroups(rankdict, nClassifiers, measure['critDist'])
		measure['ymax'] = len(measure['groups'])
	return performance


def addRanks(scores, classifiers, rankdict):
	sortedscores = sorted(zip(scores, classifiers), key=lambda pair: pair[0], reverse=True)
	tiestart = 0
	for i in range(1, len(scores)):
		if abs(sortedscores[tiestart][0] - sortedscores[i][0]) < TOL:
			# found another tie
			pass
		else:
			# tied group ends
			for j in range(tiestart, i):
				# set average ranks:
				rankdict[sortedscores[j][1]] += 1 + 1.0 * (i - 1 + tiestart) / 2
			tiestart = i
	# assign ranks to last tied group:
	for j in range(tiestart, len(scores)):
		rankdict[sortedscores[j][1]] += 1 + 1.0 * (len(scores) - 1 + tiestart) / 2


def calculateGroups(rankdict, nClassifiers, critDist):
	groups = []
	sortedClassifiers = sorted(rankdict.items(), key=lambda pair: pair[1])
	groupStart = 0
	groupEnd = 0
	for i in range(1, nClassifiers):
		if abs(sortedClassifiers[i][1] - sortedClassifiers[groupStart][1]) < critDist:
			groupEnd = i
		else:
			# group completed:
			if groupStart != groupEnd:
				groups.append({'start': sortedClassifiers[groupStart][1], 'end': sortedClassifiers[groupEnd][1],
							   'index': len(groups) + 1,
							   'classifiers': [sortedClassifiers[k][0] for k in range(groupStart, groupEnd + 1)]})
			while abs(sortedClassifiers[i][1] - sortedClassifiers[groupStart][1]) > critDist:
				# go to next start. The next useful group contains the i-th classifier and not the groupstart-th. If i-th classifier has no group, this while stops when groupstart = i"
				groupStart += 1
	# if, by the end, groupstart has not moved to the end, the last item is in one more group:
	groups.append({'start': sortedClassifiers[groupStart][1], 'end': sortedClassifiers[-1][1], 'index': len(groups) + 1,
				   'classifiers': [sortedClassifiers[k][0] for k in range(groupStart, nClassifiers)]})
	return groups


def exportComparison(export_type, data, pvalue):
	# print "At exportComparison, exporting " + export_type
	# try:
	# 	comparison = Comparison.objects.get(guid = comparison_id)
	# except:
	# 	return HttpResponseRedirect('/nocomparison/'+comparison_id+'/')
	# print "blabla", pvalue, str(request.user)
	filename = 'critDiff'  # +('' if str(request.user) == 'AnonymousUser' else '-'+str(request.user))

	file = open('exports/' + filename + '.tex', 'w+')
	with open('exports/' + filename + '.tex', 'w') as f:
		if export_type == 'pdf':
			writeheader(f)
		f.write(r"\begin{figure}")
		f.write('\n')
		f.write(r"\begin{tikzpicture}[xscale=2]")
		f.write('\n')
		# data = eval(comparison.data)
		# COMPUTE NEW VALUES
		algorithms = data[0]['names']
		nAlgorithms = len(algorithms)
		pvalue = float(pvalue.replace("-", "."))
		for measure in data:
			nDataSets = len(measure[algorithms[0]])
			measure['pvalue'] = pvalue
			measure['critDist'] = math.sqrt(nAlgorithms * (nAlgorithms + 1.0) / (6 * nDataSets)) * \
								  STUDENTIZED_RANGE_VALUES[pvalue][nAlgorithms]
			measure['groups'] = calculateGroups(dict(measure['averageRanks']), nAlgorithms, measure['critDist'])
			measure['ymax'] = len(measure['groups'])
		# END new values
		n, i = len(data[0]['averageRanks']) - 1, 1
		for alg, r in sorted(data[0]['averageRanks'], key=lambda x: x[1]):
			f.write(r"\draw (%d-%f, 0) -- (%d-%f ,-%f) -- (%d+0.2, -%f) node[anchor=west] {%s (%.1f)};" % (
			n, r - 1, n, r - 1, i * 0.5, n, i * 0.5, alg, r))
			f.write('\n')
			i += 1
		f.write(r"\draw[very thick, red] (0,0.9) -- (0,1.1);")
		f.write('\n')
		f.write(r"\draw[very thick, red] (%f, 0.9) -- (%f, 1.1);" % (data[0]['critDist'], data[0]['critDist']))
		f.write('\n')
		f.write(r"\draw[very thick, red] (0,1) -- node[anchor=south] {CD = %.2f} (%f,1);" % (
		data[0]['critDist'], data[0]['critDist']))
		f.write('\n')
		for i, item in enumerate(data[0]['groups']):
			f.write(r"\draw[very thick] (%d - %f - 0.05, -0.2*%d) -- (%d- %f + 0.05, -0.2*%d);" % (
			n, item['end'] - 1, i + 1, n, item['start'] - 1, i + 1))
			f.write('\n')
		f.write(r'\draw ' + ' -- '.join(['(%d, 0)' % i for i in range(n + 1)]) + ';')
		f.write('\n')
		for i in range(n + 1):
			f.write(r"\draw (%d,0.2) node[anchor=south] {%d} -- (%d,0);" % (i, n - i + 1, i))
			f.write('\n')
		f.write(r"\end{tikzpicture}")
		f.write('\n')
		f.write(r'\end{figure}')
		if export_type == 'pdf':
			writefooter(f)
	if export_type == 'pdf':
		with open('exports/log.txt', 'w') as logfile:
			p = Popen(['pdflatex', '-output-directory', './exports', 'exports/' + filename + '.tex'], stdout=logfile)
			p.communicate()
	suffix = '.pdf' if export_type == 'pdf' else '.tex'
	# return HttpResponseRedirect('/compare/export/' + filename + suffix)


def writeheader(f):
	f.write(r'''\documentclass{article}
\usepackage{tikz}
\usepackage{verbatim}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tikzpicture}
\begin{document}
		''')


def writefooter(f):
	f.write(r'''\end{document}''')


if __name__ == "__main__":
	# TODO run as script from command line
	pass
