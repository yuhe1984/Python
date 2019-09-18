
# coding: utf-8

# In[4]:


import multiprocessing as mp
import threading as td
import time


# In[5]:


def job(q):
    res = 0
    for i in range(100000):
        res += i+i**2+i**3
    q.put(res)    # queue


# In[6]:


def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q,))   # 单值输入加,
    p2 = mp.Process(target=job,args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multicore:',res1+res2)


# In[7]:


def normal():
    res = 0
    for _ in range(2):
        for i in range(100000):
            res += i+i**2+i**3
    print('normal:',res)


# In[8]:


def multithread():
    q = mp.Queue()
    t1 = td.Thread(target=job,args=(q,))   # 单值输入加,
    t2 = td.Thread(target=job,args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithread:',res1+res2)


# In[ ]:


if __name__ == '__main__':
    st = time.time()
    normal()
    st1 = time.time()
    print('normal time:',st1 - st)
    multithread()
    st2 = time.time()
    print('multithread:',st2 - st)
    multicore()
    st3 = time.time()
    print('multicore:',st3 - st)

