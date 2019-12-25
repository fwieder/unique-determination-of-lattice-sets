import time

def printProgressBar(iteration, total, starttime=0, decimals=1, length=50, fill='█', printEnd="\r"):
    '''
    Stolen from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    '''
    if iteration > 0 :
        timeLeft = (total-iteration)*(time.perf_counter() - starttime)/iteration 
        suffix= "   Time left: %3dm %2ds" % (timeLeft//60, timeLeft%60)
    else:
        suffix = ""
        
    percent =  ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r|%s| %s%% %s' % (bar, percent,suffix), end = printEnd)
    if iteration == total:
        print()
        