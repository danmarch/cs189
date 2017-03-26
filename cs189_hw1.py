#For CS189 at UC Berkeley
#Training a digit classifier using a soft margin SVM
#Dan March, Spring 2016

import numpy as np
from scipy import io
from sklearn import svm
from sklearn.metrics import confusion_matrix
import random
from random import shuffle
import matplotlib.pyplot as plot

def problem_one(matrix_size=0):
    sample_sizes = [100,200,500,1000,2000,5000,10000]
    returnList, cumul_answers_solver = [], []

    #Starting out the for loop for all the sample sizes
    for sample_size in sample_sizes:
        random_nums,rand_set,validation_nums,numCount1 = [],set(),[],0
        #Getting a list of random nums, sample_size long
        while numCount1 < sample_size:
            randNum1 = random.randint(0,60000-1)
            if randNum1 in rand_set:
                continue
            random_nums.append(randNum1)
            rand_set.add(randNum1)
            numCount1 += 1

        train_set = io.loadmat(file_name="data/digit-dataset/train.mat")
        train_images = np.array(train_set['train_images'])
        train_labels = np.array(list(map(int,(train_set['train_labels']))))

        #Getting random nums for the validation set
        numCount2 = 0
        while numCount2 < 10000:
            randNum2 = random.randint(0,60000-1)
            if randNum2 in rand_set:
                continue
            validation_nums.append(randNum2)
            rand_set.add(randNum2)
            numCount2 += 1

        #Creating the 10,000 validation images and answers
        validation_list = []
        for num in validation_nums:
            elm1,elm2 = train_images[:,:,num],train_labels[num]
            validation_list.append([elm1,elm2])
        validation_images = [a for a,b in validation_list]
        validation_answers = [b for a,b in validation_list]

        #Getting the sample_size number of images and labels
        images_used,labels_used = [],[]
        for num in random_nums:
            images_used.append(train_images[:,:,num])
            labels_used.append(train_labels[num])

        twodim_images = np.empty([sample_size,784])
        valid_images_used = np.empty([10000,784])

        #Converting the 3D array of images to 2D

        count1, count2 = 0,0
        for count in range(sample_size):
            index = 0
            for count2 in range(28):
                for count3 in range(28):
                    twodim_images[count][index] = \
                    images_used[count][count2][count3]
                    index += 1

        #Converting the 3D array of validation set images to 2D
        for count in range(10000):
            index = 0
            for count2 in range(28):
                for count3 in range(28):
                    valid_images_used[count][index] = \
                    validation_images[count][count2][count3]
                    index += 1

        #Training on the sample size
        solver = svm.SVC(kernel='linear')
        solver.fit(twodim_images,labels_used)
        solverOutput = (solver.predict(valid_images_used))

        #Computing error
        myIndex, myTotal = 0,0
        while myIndex < 10000:
            if solverOutput[myIndex] == validation_answers[myIndex]:
                myTotal += 1
            myIndex += 1
        accuracy = float(myTotal/10000)
        returnList.append([sample_size,accuracy])
        cumul_answers_solver.append([validation_answers,solverOutput])

    print(returnList)

    if matrix_size != 0:
        index = sample_sizes.index(matrix_size)
        cm = confusion_matrix(cumul_answers_solver[index][0],cumul_answers_solver[index][1])
        plot.figure()
        plot_cm(cm,matrix_size)
        plot.show()
    return

def plot_cm(matrix,matrix_size):
    plot.imshow(matrix, interpolation='nearest', cmap=plot.cm.Blues)
    plot.title('Confusion Matrix, '+str(matrix_size)+ ' Samples')
    plot.colorbar()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')

def problem_two(matrix_size):
    return problem_one(matrix_size)

def problem_three():
    print("Optimal C is: .0000001")
    return big_function(.0000001)

def big_function(input):
#Bad name, I know.
    train_set = io.loadmat(file_name="data/digit-dataset/train.mat")
    train_images = np.array(train_set['train_images'])
    train_labels = np.array(list(map(int,(train_set['train_labels']))))

    numCount2,rand_set,validation_nums = 0,set(),[]
    #First change
    pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7, pr8, pr9 = \
    [],[],[],[],[],[],[],[],[],[]

    #Getting 10,000 random numbers for validation set, put in validation_nums
    while numCount2 < 10000:
        randNum2 = random.randint(0,60000-1)
        if randNum2 in rand_set:
            continue
        validation_nums.append(randNum2)
        rand_set.add(randNum2)
        numCount2 += 1

    #Constructing the 10,000 images/labels for validation set.
    #validation_images has the 10,000 images in a list, and validation_answers
    #has their corresponding answers.
    validation_list = []
    for num in validation_nums:
        elm1,elm2 = train_images[:,:,num],train_labels[num]
        validation_list.append([elm1,elm2])
    validation_images = [a for a,b in validation_list]
    validation_answers = [b for a,b in validation_list]

    #Putting sublists of validation_images and validation_answers into a new
    #list, valimg_partitioned_one, for partitioning.
    #Elements take form [image,answer]
    index2 = 0
    while index2 < 10000:
        elm1, elm2 = validation_images[index2],validation_answers[index2]
        if index2 < 1000: pr0.append([elm1,elm2])
        elif index2 < 2000: pr1.append([elm1,elm2])
        elif index2 < 3000: pr2.append([elm1,elm2])
        elif index2 < 4000: pr3.append([elm1,elm2])
        elif index2 < 5000: pr4.append([elm1,elm2])
        elif index2 < 6000: pr5.append([elm1,elm2])
        elif index2 < 7000: pr6.append([elm1,elm2])
        elif index2 < 8000: pr7.append([elm1,elm2])
        elif index2 < 9000: pr8.append([elm1,elm2])
        else: pr9.append([elm1,elm2])
        index2 += 1
    valimg_partitioned_one = [pr0,pr1,pr2,pr3,pr4,pr5,pr6,pr7,pr8,pr9]

    #Run and save tests on each partition of the validation set
    #Use the 0th set, then 1st, then, ... ,then 9th set as validation set
    cumul_answers_solver,test_set_index = [],0
    while test_set_index < 10:
        cur_index,test_sets,valimg_partitioned = 0,[],[]
        while cur_index < 10:
            if test_set_index == cur_index:
                valid_set = valimg_partitioned_one[cur_index]
            else:
                test_sets.append(valimg_partitioned_one[cur_index])
            cur_index += 1

        #Unpacking the elements of each test partition into a list,
        #valimg_partitioned. Puts 9000 elements as [image,answer] into
        #valimg_partitioned.
        for lst in test_sets:
            for image_answer_key in lst:
                valimg_partitioned.append(image_answer_key)

        answers_to_tests = [b for a,b in valimg_partitioned]
        final_answers = [b for a,b in valid_set]

        twodim_images = np.empty([9000,784])
        valid_images_used = np.empty([1000,784])

        #Getting the testing images in the right format
        count1,count2 = 0,0
        for count in range(9000):
            index = 0
            for count2 in range(28):
                for count3 in range(28):
                    twodim_images[count][index] = \
                    valimg_partitioned[count][0][count2][count3]
                    index += 1

        #Also getting the validation images in the right format
        for count in range(1000):
            index = 0
            for count2 in range(28):
                for count3 in range(28):
                    valid_images_used[count][index] = \
                    valid_set[count][0][count2][count3]
                    index += 1

        #Fitting to the test images and their answers, and then testing
        #on the validation images
        solver = svm.SVC(kernel='linear',C=input)
        solver.fit(twodim_images,answers_to_tests)
        solverOutput = (solver.predict(valid_images_used))

        #Computing error
        myIndex, myTotal = 0,0
        while myIndex < 1000:
            if solverOutput[myIndex] == final_answers[myIndex]:
                myTotal += 1
            myIndex += 1
        accuracy = float(myTotal/1000)
        cumul_answers_solver.append([test_set_index,accuracy])
        test_set_index += 1

        if test_set_index == 9:
            print("Now starting to write for Kaggle.")
            test_set2 = io.loadmat(file_name="data/digit-dataset/test.mat")
            test_images = np.transpose(np.array(test_set2['test_images']))
            test_images_used = np.empty([10000,784])

            #Converting the 3D array of images to 2D
            count1,count2 = 0,0
            for count in range(10000):
                index = 0
                for count2 in range(28):
                    for count3 in range(28):
                        test_images_used[count][index] = \
                        test_images[count2][count3][count]
                        index += 1

            solverOutput = solver.predict(test_images_used)
            print("Is it a number? " + str(solverOutput[0]))
            with open("cs189_hw1_q1.csv", "w+") as int_classifier:
                writing_index = 1
                int_classifier.write("Id,Category\n")
                while writing_index < 10001:
                    int_classifier.write(str(writing_index)+","+str(solverOutput[writing_index-1])+"\n")
                    writing_index += 1

    print(cumul_answers_solver)
    sum_of_errors = 0
    for lst in cumul_answers_solver:
        sum_of_errors += 1-lst[1]
    k_error = float(sum_of_errors/10)
    print("k_error is: " + str(k_error))

def problem_four():
    print("Optimal C is: 10")
    return second_big_function(10)

def second_big_function(inputVal):
    train_set = io.loadmat(file_name="data/spam-dataset/spam_data.mat")
    training_data = np.array(list(train_set['training_data']))
    training_data = training_data.astype(int)
    training_labels = np.array(list(map(int,(train_set['training_labels'][0]))),dtype=object)

    rand_nums = [x for x in range(5170)]
    shuffle(rand_nums)
    pr0, pr1, pr2, pr3, pr4, pr5, pr6, pr7, pr8, pr9 = \
    [],[],[],[],[],[],[],[],[],[]

    index = 0
    while index < 5170:
        value = rand_nums[index]
        elm1, elm2 = training_data[value],training_labels[value]
        if index < 517: pr0.append([elm1,elm2])
        elif index < 1034: pr1.append([elm1,elm2])
        elif index < 1551: pr2.append([elm1,elm2])
        elif index < 2068: pr3.append([elm1,elm2])
        elif index < 2585: pr4.append([elm1,elm2])
        elif index < 3102: pr5.append([elm1,elm2])
        elif index < 3619: pr6.append([elm1,elm2])
        elif index < 4136: pr7.append([elm1,elm2])
        elif index < 4653: pr8.append([elm1,elm2])
        else: pr9.append([elm1,elm2])
        index += 1
    partition = [pr0,pr1,pr2,pr3,pr4,pr5,pr6,pr7,pr8,pr9]

    cumul_answers_solver,test_set_index = [],0
    while test_set_index < 10:
        cur_index,test_sets,test_set_unpacked = 0,[],[]
        while cur_index < 10:
            if test_set_index == cur_index:
                valid_set = partition[cur_index]
            else:
                test_sets.append(partition[cur_index])
            cur_index += 1

        #Unpacking the elements of each test partition into a list,
        #valimg_partitioned. Puts elements as [image,answer] into
        #valimg_partitioned.
        for lst in test_sets:
            for image_answer_key in lst:
                test_set_unpacked.append(image_answer_key)

        answers_to_tests = [b for a,b in test_set_unpacked]
        final_answers = [b for a,b in valid_set]

        #print(len(test_set_unpacked))
        #print(len(test_set_unpacked[0]))
        #print(len(test_set_unpacked[0][1]))

        #Unpacking the test emails
        twodim_data_used = np.empty([4653,32])
        valid_set_used = np.empty([517,32])
        valid_set_used = valid_set_used.astype(int)
        count1, count2 = 0,0
        for count in range(4653):
            index = 0
            for count2 in range(1):
                for count3 in range(32):
                    value = int(test_set_unpacked[count][count2][count3])
                    twodim_data_used[count][index] = value
                    #test_set_unpacked[count][count2][count3]
                    index += 1

        #Unpacking the validation emails
        for count in range(517):
            index = 0
            for count2 in range(1):
                for count3 in range(32):
                    value = int(valid_set[count][count2][count3])
                    valid_set_used[count][index] = int(value)
                    #valid_set[count][count2][count3]

        solver = svm.SVC(kernel='linear',C=inputVal)
        solver.fit(twodim_data_used,answers_to_tests)
        solverOutput = (solver.predict(valid_set_used))

        #Computing error
        myIndex, myTotal = 0,0
        while myIndex < 517:
            if solverOutput[myIndex] == final_answers[myIndex]:
                myTotal += 1
            myIndex += 1
        accuracy = float(myTotal/517)
        cumul_answers_solver.append([test_set_index,accuracy])
        test_set_index += 1

        if test_set_index == 10:
            test_set2 = io.loadmat(file_name="data/spam-dataset/spam_data.mat")
            test_data = np.array(test_set2['test_data'])

            solverOutput = solver.predict(test_data)
            with open("cs189_hw1_q4.csv", "w+") as spam_classifier:
                writing_index = 1
                spam_classifier.write("Id,Category\n")
                while writing_index < 5858:
                    spam_classifier.write(str(writing_index)+","+str(solverOutput[writing_index-1])+"\n")
                    writing_index += 1
