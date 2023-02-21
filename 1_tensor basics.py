#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch


# In[2]:


import torch


# In[3]:


x = torch.rand(3)


# In[4]:


print(x)


# In[5]:


x = torch.ones(2,2,dtype=torch.float16)


# In[6]:


print(x)


# In[7]:


x = torch.tensor([2.5,0.1])


# In[8]:


print(x)


# In[9]:


torch.cuda.is_available()


# In[10]:


x = torch.rand(2,2)
y = torch.rand(2,2)


# In[11]:


x


# In[12]:


y


# In[13]:


z = x + y


# In[14]:


z


# In[15]:


z = torch.add(x,y)


# In[16]:


z


# In[17]:


y.add_(x)


# In[18]:


z = x-y


# In[19]:


z


# In[20]:


z = torch.sub(x,y)


# In[21]:


z


# In[22]:


y.sub(x)


# In[23]:


z = x*y
z


# In[24]:


z = torch.mul(x,y)


# In[25]:


z


# In[26]:


z = x/y
z


# In[27]:


z = torch.div(x,y)
z


# In[28]:


x = torch.rand(5,3)
x


# In[29]:


x[:,0]


# In[30]:


import numpy as np


# In[31]:


a = torch.ones(5)


# In[32]:


a


# In[33]:


b = a.numpy()


# In[34]:


b


# In[35]:


a.add_(1)


# In[36]:


a = np.ones(5)
a


# In[37]:


b = torch.from_numpy(a)
b


# In[38]:


a += 1
a


# In[39]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y
    z = z.to("cpu")


# In[40]:


x = torch.ones(5,requires_grad=True)
x


# In[41]:


# AutoGrad


# In[42]:


import torch
x = torch.rand(3)
print(x)


# In[43]:


x = torch.rand(3,requires_grad=True)


# In[44]:


print(x)


# In[45]:


y = x+2
print(y)


# In[46]:


print(x.grad)


# In[47]:


z = y*y*2
print(z)


# In[48]:


z = z.mean()


# In[49]:


print(z)


# In[50]:


z.backward()


# In[51]:


print(x.grad)


# In[52]:


# prevent tracking gradient


# In[54]:


x.requires_grad_(False)


# In[55]:


x.detach()


# In[56]:


#with torch.no_grad():


# In[57]:


import torch
x = torch.randn(3,requires_grad=True)


# In[58]:


x


# In[59]:


x.requires_grad_(False)


# In[60]:


y = x.detach()
y


# In[61]:


with torch.no_grad():
    y = x+2
    print(y)


# In[62]:


# example 1 


# In[63]:


import torch
weights = torch.ones(4,requires_grad=True)
print(weights)


# In[64]:


for epoch in range(1):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
    print(weights)


# In[65]:


# through optimizer


# In[66]:


import torch
weights = torch.ones(4,requires_grad=True)
optimizer = torch.optim.SGD([weights],lr=0.01)
optimizer.step()
optimizer.zero_grad


# In[67]:


# Back Propagation


# In[68]:


import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0,requires_grad=True)


# In[69]:


# Forward Pass


# In[70]:


y_hat = w*x
loss = (y_hat-y)**2
print(loss)


# In[71]:


# Backward Pass


# In[72]:


loss.backward()
print(w.grad)


# In[73]:


import tensorflow as tf


# In[74]:


tf.__version__


# In[75]:


torch.version.cuda


# In[76]:


get_ipython().system('nvid')


# In[ ]:




