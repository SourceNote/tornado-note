# coding:utf-8
"""各种工具函数/类.

本模块是Tornado在内部使用。不必期望这里定义的函数何磊可以再其他的应用中使用。

本模块对外公开的部分是`Configurable`类及它的`~Configurable.configure` 方法。
这部分已经成为它的子类`.AsyncHTTPClient`, `.IOLoop`,和 `.Resolver`接口的一部分

"""

from __future__ import absolute_import, division, print_function

import array
import atexit
import os
import re
import sys
import zlib

PY3 = sys.version_info >= (3,)

if PY3:
    xrange = range

# inspect.getargspec() raises DeprecationWarnings in Python 3.5.
# The two functions have compatible interfaces for the parts we need.
if PY3:
    from inspect import getfullargspec as getargspec
else:
    from inspect import getargspec

# Aliases for types that are spelled differently in different Python
# versions. bytes_type is deprecated and no longer used in Tornado
# itself but is left in case anyone outside Tornado is using it.
bytes_type = bytes
if PY3:
    unicode_type = str
    basestring_type = str
else:
    # unicode 和 basestring 在 py3 不存在，所以默认flake8.
    unicode_type = unicode  # noqa
    basestring_type = basestring  # noqa

try:
    import typing  # noqa
    from typing import cast

    _ObjectDictBase = typing.Dict[str, typing.Any]
except ImportError:
    _ObjectDictBase = dict


    def cast(typ, x):
        return x
else:
    # 导入在类型注释中需要的部分.
    import datetime  # noqa
    import types  # noqa
    from typing import Any, AnyStr, Union, Optional, Dict, Mapping  # noqa
    from typing import Tuple, Match, Callable  # noqa

    if PY3:
        _BaseString = str
    else:
        _BaseString = Union[bytes, unicode_type]

try:
    from sys import is_finalizing
except ImportError:
    # Emulate it
    def _get_emulated_is_finalizing():
        L = []
        atexit.register(lambda: L.append(None))

        def is_finalizing():
            # 这里不引用任何全局属性的东西
            # Not referencing any globals here
            return L != []

        return is_finalizing


    is_finalizing = _get_emulated_is_finalizing()


class ObjectDict(_ObjectDictBase):
    """使一个字典和对象一样，可以通过属性风格进行访问.
    """

    def __getattr__(self, name):
        # 类型: (str) -> 任意
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        # 类型: (str, 任意) -> None
        self[name] = value


class GzipDecompressor(object):
    """流gzip解压缩.

    接口类似于 `zlib.decompressobj` (没有一些可选参数，
     但是他能理解处理gzip的头部和校验和.）
    """

    def __init__(self):
        # 魔法函数使得 zlib 模块能理解 gzip 头部
        # http://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib
        # 在 cpython 和 pypy 中可以工作, 但是jython中不可以.
        self.decompressobj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def decompress(self, value, max_length=None):
        # 类型: (bytes, Optional[int]) -> bytes
        """解压缩一个chunk, 返回一个新的可用数据
        一些数据可能会被缓冲以待之后处理;当没有更多数据输入的时候，
        为了保证所有的数据都被处理，必须要调用`flush`

        如果提供了 ``max_length`` , 一些输入数据可能会被遗留在 ``unconsumed_tail``;
        你必须遍历这个数据并将其传递给一个之后要调用的函数，若其非空，就进行解压操作
        """
        return self.decompressobj.decompress(value, max_length)

    @property
    def unconsumed_tail(self):
        # 类型: () -> bytes
        """返回遗留下来的未被处理消耗尽的部分
        """
        return self.decompressobj.unconsumed_tail

    def flush(self):
        # 类型: () -> bytes
        """Return any remaining buffered data not yet returned by decompress.

        同时检查如输入被删节的错误
        No other methods may be called on this object after `flush`.
        """
        return self.decompressobj.flush()


def import_object(name):
    # type: (_BaseString) -> Any
    """通过名字导入一个对象.

    import_object('x') 等同于 'import x'.
    import_object('x.y.z') 等同于 'from x.y import z'.

    >>> import tornado.escape
    >>> import_object('tornado.escape') is tornado.escape
    True
    >>> import_object('tornado.escape.utf8') is tornado.escape.utf8
    True
    >>> import_object('tornado') is tornado
    True
    >>> import_object('tornado.missing_module')
    Traceback (most recent call last):
        ...
    ImportError: No module named missing_module
    """
    if not isinstance(name, str):
        # python 2 需要一个 byte 字符串
        name = name.encode('utf-8')
    if name.count('.') == 0:
        return __import__(name, None, None)

    parts = name.split('.')
    obj = __import__('.'.join(parts[:-1]), None, None, [parts[-1]], 0)
    try:
        return getattr(obj, parts[-1])
    except AttributeError:
        raise ImportError("No module named %s" % parts[-1])


# 占位 (稍后进行真正的类型检查).
def raise_exc_info(exc_info):
    # 类型: (Tuple[type, BaseException, types.TracebackType]) -> None
    pass


def exec_in(code, glob, loc=None):
    # 类型: (任意, Dict[str, 任意], 可选[Mapping[str, 任意]]) -> 任意
    if isinstance(code, basestring_type):
        # exec(string) 继承调用者的future imports; 先编译字符串来阻止它
        code = compile(code, '<string>', 'exec', dont_inherit=True)
    exec (code, glob, loc)


if PY3:
    exec ("""
def raise_exc_info(exc_info):
    try:
        raise exc_info[1].with_traceback(exc_info[2])
    finally:
        exc_info = None

""")
else:
    exec ("""
def raise_exc_info(exc_info):
    raise exc_info[0], exc_info[1], exc_info[2]
""")


def errno_from_exception(e):
    # 类型: (BaseException) -> 可选[int]
    """提供异常对象的错误代码.

    There are cases that the errno attribute was not set so we pull
    the errno out of the args but if someone instantiates an Exception
    without any args you will get a tuple error. So this function
    abstracts all that behavior to give you a safe way to get the
    errno.
    """

    if hasattr(e, 'errno'):
        return e.errno  # type: ignore
    elif e.args:
        return e.args[0]
    else:
        return None


# Tacey：将字串转为不可变集合
_alphanum = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _re_unescape_replacement(match):
    # 类型: (Match[str]) -> str
    group = match.group(1)
    if group[0] in _alphanum:
        raise ValueError("不能反转义 '\\\\%s'" % group[0])
    return group


_re_unescape_pattern = re.compile(r'\\(.)', re.DOTALL)


def re_unescape(s):
    # 类型: (str) -> str
    """反转义`re.escape`转义的字符串.

    May raise ``ValueError`` for regular expressions which could not
    have been produced by `re.escape` (for example, strings containing
    ``\d`` cannot be unescaped).

    .. versionadded:: 4.4
    """
    return _re_unescape_pattern.sub(_re_unescape_replacement, s)


class Configurable(object):
    """可配置接口基类.

    A configurable interface is an (abstract) class whose constructor
    acts as a factory function for one of its implementation subclasses.
    The implementation subclass as well as optional keyword arguments to
    its initializer can be set globally at runtime with `configure`.

    By using the constructor as the factory method, the interface
    looks like a normal class, `isinstance` works as usual, etc.  This
    pattern is most useful when the choice of implementation is likely
    to be a global decision (e.g. when `~select.epoll` is available,
    always use it instead of `~select.select`), or when a
    previously-monolithic class has been split into specialized
    subclasses.

    Configurable subclasses must define the class methods
    `configurable_base` and `configurable_default`, and use the instance
    method `initialize` instead of ``__init__``.
    """
    __impl_class = None  # type: type
    __impl_kwargs = None  # type: Dict[str, Any]

    def __new__(cls, *args, **kwargs):
        base = cls.configurable_base()
        init_kwargs = {}
        if cls is base:
            impl = cls.configured_class()
            if base.__impl_kwargs:
                init_kwargs.update(base.__impl_kwargs)
        else:
            impl = cls
        init_kwargs.update(kwargs)
        instance = super(Configurable, cls).__new__(impl)
        # initialize vs __init__ chosen for compatibility with AsyncHTTPClient
        # singleton magic.  If we get rid of that we can switch to __init__
        # here too.
        instance.initialize(*args, **init_kwargs)
        return instance

    @classmethod
    def configurable_base(cls):
        # type: () -> Any
        # TODO: This class needs https://github.com/python/typing/issues/107
        # to be fully typeable.
        """Returns the base class of a configurable hierarchy.

        This will normally return the class in which it is defined.
        (which is *not* necessarily the same as the cls classmethod parameter).
        """
        raise NotImplementedError()

    @classmethod
    def configurable_default(cls):
        # type: () -> type
        """Returns the implementation class to be used if none is configured."""
        raise NotImplementedError()

    def initialize(self):
        # type: () -> None
        """Initialize a `Configurable` subclass instance.

        Configurable classes should use `initialize` instead of ``__init__``.

        .. versionchanged:: 4.2
           Now accepts positional arguments in addition to keyword arguments.
        """

    @classmethod
    def configure(cls, impl, **kwargs):
        # type: (Any, **Any) -> None
        """Sets the class to use when the base class is instantiated.

        Keyword arguments will be saved and added to the arguments passed
        to the constructor.  This can be used to set global defaults for
        some parameters.
        """
        base = cls.configurable_base()
        if isinstance(impl, (str, unicode_type)):
            impl = import_object(impl)
        if impl is not None and not issubclass(impl, cls):
            raise ValueError("Invalid subclass of %s" % cls)
        base.__impl_class = impl
        base.__impl_kwargs = kwargs

    @classmethod
    def configured_class(cls):
        # type: () -> type
        """Returns the currently configured class."""
        base = cls.configurable_base()
        if cls.__impl_class is None:
            base.__impl_class = cls.configurable_default()
        return base.__impl_class

    @classmethod
    def _save_configuration(cls):
        # type: () -> Tuple[type, Dict[str, Any]]
        base = cls.configurable_base()
        return (base.__impl_class, base.__impl_kwargs)

    @classmethod
    def _restore_configuration(cls, saved):
        # type: (Tuple[type, Dict[str, Any]]) -> None
        base = cls.configurable_base()
        base.__impl_class = saved[0]
        base.__impl_kwargs = saved[1]


class ArgReplacer(object):
    """替换``args, kwargs`` 对的值.

    Inspects the function signature to find an argument by name
    whether it is passed by position or keyword.  For use in decorators
    and similar wrappers.
    """

    def __init__(self, func, name):
        # type: (Callable, str) -> None
        self.name = name
        try:
            self.arg_pos = self._getargnames(func).index(name)
        except ValueError:
            # 不是一个位置参数
            self.arg_pos = None

    def _getargnames(self, func):
        # type: (Callable) -> List[str]
        try:
            return getargspec(func).args
        except TypeError:
            if hasattr(func, 'func_code'):
                # Cython-generated code has all the attributes needed
                # by inspect.getargspec, but the inspect module only
                # works with ordinary functions. Inline the portion of
                # getargspec that we need here. Note that for static
                # functions the @cython.binding(True) decorator must
                # be used (for methods it works out of the box).
                code = func.func_code  # type: ignore
                return code.co_varnames[:code.co_argcount]
            raise

    def get_old_value(self, args, kwargs, default=None):
        # type: (List[Any], Dict[str, Any], Any) -> Any
        """Returns the old value of the named argument without replacing it.

        Returns ``default`` if the argument is not present.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            return args[self.arg_pos]
        else:
            return kwargs.get(self.name, default)

    def replace(self, new_value, args, kwargs):
        # type: (Any, List[Any], Dict[str, Any]) -> Tuple[Any, List[Any], Dict[str, Any]]
        """Replace the named argument in ``args, kwargs`` with ``new_value``.

        Returns ``(old_value, args, kwargs)``.  The returned ``args`` and
        ``kwargs`` objects may not be the same as the input objects, or
        the input objects may be mutated.

        If the named argument was not found, ``new_value`` will be added
        to ``kwargs`` and None will be returned as ``old_value``.
        """
        if self.arg_pos is not None and len(args) > self.arg_pos:
            # The arg to replace is passed positionally
            old_value = args[self.arg_pos]
            args = list(args)  # *args is normally a tuple
            args[self.arg_pos] = new_value
        else:
            # The arg to replace is either omitted or passed by keyword.
            old_value = kwargs.get(self.name)
            kwargs[self.name] = new_value
        return old_value, args, kwargs


def timedelta_to_seconds(td):
    # 类型: (datetime.timedelta) -> float
    """等同于td.total_seconds() (python 2.7)."""
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / float(10 ** 6)


def _websocket_mask_python(mask, data):
    # 类型: (bytes, bytes) -> bytes
    """Websocket掩码函数.

    `mask` 是一个长度为4的 `bytes`对象 ; `data` 是一个任意长度的 `bytes` 对象.
    Returns a `bytes` object of the same length as `data` with the mask applied
    as specified in section 5.3 of RFC 6455.

    此纯Python实现版本可以替换为更优化的版本
    """
    mask_arr = array.array("B", mask)
    unmasked_arr = array.array("B", data)
    for i in xrange(len(data)):
        unmasked_arr[i] = unmasked_arr[i] ^ mask_arr[i % 4]
    if PY3:
        # tostring在py32是不提倡的。它现在已经被移除,
        # 但是由于我们的测试中会提示为不提倡警告
        # 所以我们要使用正确的那个.
        return unmasked_arr.tobytes()
    else:
        return unmasked_arr.tostring()


if (os.environ.get('TORNADO_NO_EXTENSION') or
            os.environ.get('TORNADO_EXTENSION') == '0'):
    # 这些环境变量的存在便于进行性能对比，不保证将来会对此进行支持.
    _websocket_mask = _websocket_mask_python
else:
    try:
        from tornado.speedups import websocket_mask as _websocket_mask
    except ImportError:
        if os.environ.get('TORNADO_EXTENSION') == '1':
            raise
        _websocket_mask = _websocket_mask_python


def doctests():
    import doctest
    return doctest.DocTestSuite()
