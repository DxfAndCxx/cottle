
============================
Cottle: Python Web Framework
============================
这个项目是基于bottle 进行开发的. 开发之后的bottle 可能与bottle 的原项目的区别很
大. 

之所以基于bottle 进行开发是因为我很喜欢bottle. bottle的代码也是很好的. 我从中学
习到了很多东西. 一些处理在我看来非常好. 

我开发项目的另一个原因是我试用了webpy. 这个项目也是很有特色的. 但是我相对而言在
代码上我更加喜欢bottle.
webpy 吸引我的地方在于使用类做为回调. 

对于url的处理使用urls, bottle使用decorator的方式是一具亮点, 但是如果
开发的量大一些, 就会给人很乱的感觉. 很不方便管理. 

另一点在于使用函数做为回调不能体现出多个不同的url之间的关系.
一些请求之间有可能是相关的.

另一个好处是可以与restful 进行很好的配合. restful 在我看来有很重的OO 的影子.


bottle的只有一个文件的优势是以代码维护为代架的.
如果使用zip 进行发布一个文件的优势也是一样的.

webpy 中让我比较头的问题在于对于请求信息的处理. 这一点在bottle 中做得很好.
request 类是很让我满意的. 这里有另一个问题. 在处理请求的时候必然有一些由
client 带来的信息. 这些信息bottle与web 的处理是一样的都是调用其它的
接口完成. 但是如果使用webpy的思路我们可以把一些信息直接放到回调类上.
这样进行处理的时候会更加方便. 

我想把这两个框架结合一下. 


使用类进行回调的另一个好处是可以在处理的过程中把一些信息结合的更好一些.
比如: 在调用类的GET方法之前可以把url 的请求参数与post的请求参数设置成
对象的属性. 顺GET 方法中可以很方便地得到这些参数.

在处理一个GET 请求之前可能有一些要先进行处理的东西. 这些东西如权限的控制. 
这种控制一般是针对于一个对象的. 在restful 里url 是一个对象. 在这里一个url 对应于
一个类. 那么就是说对于个权限的控制是应该是在这个类里完成的. 过去我使用bottle
在函数的开头加上这些处理. 但是如果可以把这些处理放到的类的Before 方法中. 可能更
好一些.




License
-------

.. __: https://github.com/defnull/bottle/raw/master/LICENSE

Code and documentation are available according to the MIT License (see LICENSE__).

The Cottle logo however is *NOT* covered by that license. It is allowed to use the logo as a link to the bottle homepage or in direct context with the unmodified library. In all other cases please ask first.
