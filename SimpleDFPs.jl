__precompile__()
module SimpleDFPs

    import Base: ==, +, -, *, /, \, inv, ^, <, <=, <<, >>, %, ÷
    import Base: mod, fld, cld, floor, ceil, sqrt, cbrt, fma, sincos, factorial
    import Base: round, decompose, ldexp, frexp
    import Base: nextfloat, prevfloat, eps

    using SpecialFunctions

    # exporting all the functions
    export DFP, DFP_getDefaultPrecision, DFP_setDefaultPrecision, DFP_setAllDefaultPrecision
    export DFP_getShowStyle, DFP_setShowStyle
    export DFP_temporarily_setDefaultPrecision
    export DFP_fromString
    export FromDigits, IntegerDigits, @d_str
    export call, call_viaFloat64
    export DFP_listbuilder, DFP_normalise, DFP_relativeEpsilon
    export DFP_lengthMantissa, DFP_mapBinaryFuncToList
    export DFP_convertOtherFloatToCommonString
    export DFP_toFortranString, DFP_toCommonString, DFP_toIntegerString
    export DFP_toShortCommonString, DFP_toTinyCommonString, DFP_toNumCommonString
    export DFP_toExplainationString, DFP_toNdigitString
    export DFP_toCanonicalString, DFP_toMantissaString
    export DFP_toHardFloatingPointString, DFP_toMathematicaString
    export DFP_toFloat64
    export DFP_isZero, DFP_isOne, DFP_isMinusOne, DFP_isDivByZero
    export DFP_isOtherError, DFP_isError
    export DFP_isNormalNumber, DFP_isSpecialNumber, DFP_isOrdinaryError
    export DFP_isNaN, DFP_isInf, DFP_isNegInf
    export DFP_kind, DFP_lhsrhs_kind
    export DFP_isPositive, DFP_isNegative
    export DFP_isEqual, DFP_isInteger, DFP_isFraction
    export DFP_round
    export DFP_roundup, DFP_rounddown
    export DFP_getPrecision, DFP_setPrecision
    export DFP_createZero, DFP_createOne,DFP_createHalf
    export DFP_0, DFP_1, DFP_half
    export DFP_createOneTenth, DFP_createMinusOne
    export DFP_createTwo, DFP_createThree, DFP_createFour, DFP_createFive
    export DFP_2, DFP_3, DFP_4, DFP_5
    export DFP_createEight, DFP_createTen, DFP_createFifteen
    export DFP_8, DFP_10, DFP_15
    export DFP_createThirty, DFP_createFortyFive, DFP_createSixty
    export DFP_30, DFP_45, DFP_60
    export DFP_createSixtyFour, DFP_createNinety, DFP_createOneEighty
    export DFP_64, DFP_90, DFP_180
    export DFP_createTwoSeventy, DFP_createThreeSixty
    export DFP_270, DFP_360
    export DFP_createDivByZeroError, DFP_createError
    export DFP_createInfError, DFP_createNegInfError, DFP_createNaNError
    export DFP_abs, DFP_doublePrecision
    export oda, mdaOrig, mda, ninecomplement
    export DFP_genericAdd, DFP_add
    export DFP_genericSub, DFP_sub
    export DFP_genericMul, DFP_mul
    export DFP_genericDiv, DFP_div
    export DFP_power, DFP_nroot, nroot
    export DFP_compare
    export DFP_genericFma, DFP_fma
    export DFP_integer, DFP_fraction
    export DFP_leftShift, DFP_rightShift
    export DFP_leftCircularShift, DFP_rightCircularShift
    export DFP_euclidInteger, DFP_euclidFraction
    export DFP_ceiling, cld_rem, euclid_div, euclid_rem, div_rem
    export DFP_Pow2, DFP_Pow4, DFP_Pow8, DFP_Pow10
    export DFP_getUnitDigit, MantissaStringInScientificFormat
    export MantissaStringInLog10Format, takelog10ToIpfType
    export straightToIpfType, takeexp10OfIpfType
    export DFP_altSqrt, DFP_altLog10
    export DFP_lambertW, DFP_lambertWsb, DFP_lambertW_demo
    export DFP_log10Gamma, DFP_log10Factorial, DFP_Perm, DFP_Comb
    export DFP_sideStep, DFP_sgn, DFP_sign
    export DFP_forwardEpsilon, DFP_backwardEpsilon, DFP_bidirectionalEpsilon
    export DFP_randomDFP, DFP_randomDFPinteger
    export DFP_randomDFPbetweenZeroandOne
    export DFP_randomDFPbetweenZeroandOne_likeModel
    export DFP_randomDFPbetweenZeroandOneSafe
    export DFP_randomDFPbetweenZeroandOneSafe_likeModel
    export DFP_derivative, DFP_firstsecondDerivative
    export DFP_firstsecondthirdDerivative
    export DFP_derivative_multivariate
    export DFP_derivative_multifunc_multivariate
    export DFP_grad, DFP_jacobian, DFP_hessian, DFP_norm
    export Create_MathematicaLine
    export DFP_convertVecVectoString



## Type Definitions

    # Define our new Type
    IntegerOrSymbol = Union{Integer,Symbol}

    struct DFP{N} <: AbstractFloat
        s::Int8
        expo::BigInt
        m::Array{Int8,1}
    end

    # Integer Plus Fraction is used to perform calculations
    # with base 10 logarithmns
    struct IpfType
        int::BigInt
        d::DFP
    end

## Macro definitions

    # Allow d"3.4" to call DFP("3.4")
    macro d_str(s)
        DFP(s)
    end

## Constant definitions

    # set the version number here
    const DFP_VERSION = v"1.3.2"

    const GUARD_DIGITS = 4

    # Use the Ref hack to speed up DEFAULT_PRECISION
    # to get the value use DEFAULT_PRECISION[][Base.Threads.threadid()]
    # to set the value use DEFAULT_PRECISION[][Base.Threads.threadid()] = newvalue
    const DEFAULT_PRECISION = Ref(Int64[])

    # Controls how Base.show displays a DFP object
    # The valid values are
    # 1.  :default  or  :canon         for canonical string output
    # 2.  :normal   or  :string        for complete string output
    # 3.  :short    or  :shortstring   for short string output
    # 4.  :tiny     or  :tinystring    for tiny string output
    # 5.  <num>                        for <num> digits string output
    const SHOW_STYLE = Ref(IntegerOrSymbol[])

    const ErrorMessage = Dict{Int64,String}(
    1 => "Error 1 : Improperly formed Precision Decimal Floating Point",
    2 => "Error 2 : Precision of LHS does not match Precision of RHS",
    3 => "Error 3 : Domain error as the result is complex",
    4 => "Error 4 : Invalid root, taking the zeroth root of non-one real number",
    5 => "Error 5 : Taking the zeroth root of One results in any number as an equally valid result",
    6 => "Error 6 : Requirement error as the restriction lowerbound < middlepoint < upperbound has not being met",
    7 => "Inf",
    8 => "-Inf",
    9 => "NaN"
    )

    # Initialise the module with __init__ function
    function __init__()
        # Set the local DEFAULT_PRECISION for each thread to 16
        resize!(DEFAULT_PRECISION[], Base.Threads.nthreads())
        fill!(DEFAULT_PRECISION[],16)
        # Set the local SHOW_STYLE for each thread to :default
        resize!(SHOW_STYLE[], Base.Threads.nthreads())
        fill!(SHOW_STYLE[],:default)
    end

## Rule definitions

    # Identity rule
    DFP(x::DFP) = x
    DFP(x::DFP,prec) = DFP_setPrecision(x,prec)

    # First the promotion rules
    Base.promote_rule(::Type{DFP{N}}, ::Type{<:AbstractFloat}) where {N} = DFP{N}
    Base.promote_rule(::Type{DFP{N}}, ::Type{<:Integer}) where {N} = DFP{N}
    Base.promote_rule(::Type{DFP{N}}, ::Type{Rational{T}}) where {N,T} = DFP{N}
    Base.promote_rule(::Type{BigFloat}, ::Type{DFP{N}}) where {N} = DFP{N}

    # construction rules
    DFP{N}(num::Rational) where {N} = DFP{N}(numerator(num))/DFP{N}(denominator(num))

    DFP{N}(x::Bool) where {N} = x ? DFP{N}(1) : DFP{N}(0)
    DFP{N}(num::Real) where {N} = parse(DFP{N}, string(num))

    # Convert a number to a generic DFP with default precision
    DFP(num::Rational) = DFP(numerator(num))/DFP(denominator(num))

    DFP(x::Bool) = x ? DFP(1) : DFP(0)
    DFP(num::Real) = parse(DFP, string(num))

    # Convert a number to a DFP with specific precision
    DFP(num::Real,prec) = DFP(string(num),prec)

    # Basic definition of DFP
    DFP(s::Integer, e::Integer, m::Array{Int64,1}) = DFP{length(m)}(Int8(s), BigInt(e), append!(Int8[], m))

    DFP(s::Integer, e::Integer, m::Array{Any,1}) = DFP{length(m)}(Int8(s), BigInt(e), append!(Int8[], m))

    # Convertion to float
    Base.float(x::DFP) = DFP_toFloat64(x)
    Base.Float64(x::DFP) = DFP_toFloat64(x)

    # Convertion to integer
    Base.Int64(x::DFP) = parse(Int64,DFP_toIntegerString(round(x)))
    Base.BigInt(x::DFP) = parse(BigInt,DFP_toIntegerString(round(x)))

    # parsing rule
    function Base.parse(::Type{DFP{N}}, str::AbstractString) where {N}
        return DFP_setPrecision(DFP_fromString(str),N)
    end

    function Base.parse(::Type{DFP}, str::AbstractString)
        return DFP_setPrecision(DFP_fromString(str),DEFAULT_PRECISION[][Base.Threads.threadid()])
    end

    # Zero/one value
    Base.zero(::Type{DFP{N}}) where {N} = DFP_createZero(N)
    Base.one(::Type{DFP{N}}) where {N} = DFP_createOne(N)
    Base.string(x::DFP) = DFP_toCommonString(x)

    # Base Boolean value
    Base.signbit(x::DFP) = Bool(x.s)

    # Base queries
    Base.iszero(x::DFP) = DFP_isZero(x)
    Base.isone(x::DFP) = DFP_isOne(x)
    Base.isinteger(x::DFP) = DFP_isInteger(x)
    Base.isnan(x::DFP) = DFP_isNaN(x)
    Base.isinf(x::DFP) = DFP_isInf(x) || DFP_isNegInf(x)

    # Basic arithmetic functions

    ==(x::DFP, y::DFP) = DFP_isEqual(x,y)

    +(x::DFP{N}, y::DFP{N}) where {N} = DFP_add(x,y)

    -(x::DFP{N}) where {N} = DFP_isError(x) ? x : DFP{N}((x.s+1)%2, x.expo, x.m)

    -(x::DFP{N}, y::DFP{N}) where {N} = DFP_sub(x,y)

    *(x::DFP{N}, y::DFP{N}) where {N} = DFP_mul(x,y)

    /(x::DFP{N}, y::DFP{N}) where {N} = DFP_div(x,y)

    \(x::DFP{N}, y::DFP{N}) where {N} = DFP_div(y,x)

    inv(x::DFP{N}) where {N} = DFP_div(DFP_createOne(N),x)

    ^(x::DFP, y::Integer) = DFP_power(x,y)

    ^(x::DFP{N}, y::DFP{N}) where {N} = DFP_power(x,y)

    <(x::DFP{N}, y::DFP{N}) where {N} = DFP_compare(x,y) == -1

    <=(x::DFP{N}, y::DFP{N}) where {N} = DFP_compare(x,y) <= 0

    <<(x::DFP, n::Integer) = DFP_leftShift(x,n)

    >>(x::DFP, n::Integer) = DFP_rightShift(x,n)

    ÷(x::DFP{N},y::DFP{N}) where {N} = DFP_integer( x/y )

    %(x::DFP{N},y::DFP{N}) where {N} = x - fld(x,y) * y

#=
    -4 mod 4 == 0    -4 mod1 4 == 4
    -3 mod 4 == 1    -3 mod1 4 == 1
    -2 mod 4 == 2    -2 mod1 4 == 2
    -1 mod 4 == 3    -1 mod1 4 == 3
    0 mod 4 == 0    0 mod1 4 == 4
    1 mod 4 == 1    1 mod1 4 == 1
    2 mod 4 == 2    2 mod1 4 == 2
    3 mod 4 == 3    3 mod1 4 == 3
    4 mod 4 == 0    4 mod1 4 == 4

    -4 mod -4 == 0    -4 mod1 -4 == -4
    -3 mod -4 == -3    -3 mod1 -4 == -3
    -2 mod -4 == -2    -2 mod1 -4 == -2
    -1 mod -4 == -1    -1 mod1 -4 == -1
    0 mod -4 == 0     0 mod1 -4 == -4
    1 mod -4 == -3    1 mod1 -4 == -3
    2 mod -4 == -2    2 mod1 -4 == -2
    3 mod -4 == -1    3 mod1 -4 == -1
    4 mod -4 == 0    4 mod1 -4 == -4
=#
    # More basic Mathematical functions

    mod(x::DFP{N},y::DFP{N}) where {N} = x - fld(x,y) * y

    fld(x::DFP{N},y::DFP{N}) where {N} = DFP_euclidInteger( x/y )

    cld(x::DFP{N},y::DFP{N}) where {N} = DFP_ceiling( x / y )

    cld_rem(x::DFP{N},y::DFP{N}) where {N} = x - cld(x,y) * y

    floor(x::DFP{N}) where {N} = DFP_euclidInteger(x)

    ceil(x::DFP{N}) where {N} = DFP_ceiling(x)

    function euclid_div(x::DFP{N},y::DFP{N}) where {N}
        if y.s == 0
            return  DFP_euclidInteger( x / y )
        end
        if x.s == 0
            return DFP_integer( x / y )
        else
            return DFP_ceiling( x / y )
        end
    end

    euclid_rem(x::DFP{N},y::DFP{N}) where {N} = x - euclid_div(x,y) * y

    div_rem(x::DFP{N},y::DFP{N}) where {N} = x - div(x,y) * y

    # Now extend the Base mathematical functions

    sqrt(x::DFP{N}) where {N} = call(sqrt,x)

    cbrt(x::DFP{N}) where {N} = call(cbrt,x)

    fma(a::DFP{N},b::DFP{N},c::DFP{N}) where {N} = DFP_fma(a,b,c)

    sincos(x::DFP{N}) where {N} = (sin(x), cos(x))

    factorial(a::DFP{N}) where {N} = gamma(a + DFP_createOne(N))

    function decompose(x::DFP)::Tuple{BigInt, BigInt, Int}
        kind = DFP_kind(x)
        " return 0, 0, 0 if NaN or Error"
        (kind == 3 || kind == 4) && return 0, 0, 0
        " return 1, 0, 0 if +Inf"
        kind == 1 && return 1, 0, 0
        " return -1, 0, 0 if -Inf"
        kind == 2 && return -1, 0, 0
        s = DFP_toMantissaString(x)
        s = parse(BigInt,s)
        d = x.s == 0 ? 1 : -1
        s, x.expo, d
    end

    function ldexp(x::DFP,n::Int64)
        prec = DFP_getPrecision(x)
        newprec = prec + GUARD_DIGITS
        newx = DFP_setPrecision(x,newprec)
        return DFP_setPrecision(newx * DFP_2(newprec)^n ,prec)
    end

    function ldexp(x::DFP{N},n::DFP{N}) where {N}
        newprec = N + GUARD_DIGITS
        newx = DFP_setPrecision(x,newprec)
        newn = DFP_setPrecision(n,newprec)
        return DFP_setPrecision(newx * DFP_2(newprec)^newn ,N)
    end

    function frexp(x::DFP)
        prec  = DFP_getPrecision(x)
        newprec = prec + GUARD_DIGITS
        negflag = DFP_isNegative(x)
        newx = DFP_setPrecision(x,newprec)
        if negflag
            newx = -newx
        end
        two = DFP_2(newprec)
        b = log(newx) / log(two)
        n = DFP_integer(b) + DFP_1(newprec)
        if DFP_isZero(newx)
            f_result = DFP_createZero(prec)
            n_result = DFP_createZero(prec)
        else
            f=newx/(two^n)
            f_result = DFP_setPrecision(f,prec)
            n_result = DFP_setPrecision(n,prec)
            if f_result == DFP_1(prec)
                f_result = DFP_half(prec)
                n_result = n_result + DFP_1(prec)
            end
        end
        if negflag
            f_result = -f_result
        end
        return(f_result,n_result)
    end

    # Use metaprogramming to define Log functions
    for func in [:log, :log2, :log10, :log1p, :exp, :exp2, :exp10, :expm1]
       expr = :( Base.$func(x::DFP{N}) where {N} = call($func,x) )
       eval(expr)
    end

    # Use metaprogramming to define Trigonometry functions
    ListOfSingleArgTrigFuncs = [ :sin, :cos, :tan,
    :sind, :cosd, :tand, :csc, :sec, :cot, :cscd,
    :secd, :cotd, :asin, :acos, :atan, :asind, :acosd,
    :atand, :acsc, :asec, :acot, :acscd, :asecd, :acotd,
    :sinh, :cosh, :tanh, :csch, :sech, :coth, :asinh,
    :acosh, :atanh, :acsch, :asech, :acoth,
    :sinpi, :cospi, :sinc, :cosc, :deg2rad, :rad2deg
    ]

    for func in ListOfSingleArgTrigFuncs
       expr = :( Base.Math.$func(x::DFP{N}) where {N} = call($func,x) )
       eval(expr)
    end

    for func in [:atan, :atand, :hypot, :log]
       expr = :( Base.Math.$func(y::DFP{N},x::DFP{N}) where {N} = call($func,y,x) )
       eval(expr)
    end

    # Use metaprogramming to define Special functions
    ListOfSingleArgSpecialFuncs = [ :erfinv,
    :erfcinv, :erfi, :erfcx, :dawson, :sinint, :cosint ]

    for func in ListOfSingleArgSpecialFuncs
       expr = :( SpecialFunctions.$func(x::DFP{N}) where {N} = call_viaFloat64($func,x) )
       eval(expr)
    end

    # Use metaprogramming to define Special functions
    ListOfSingleArgSpecialFuncs = [ :erf, :erfc, :digamma,
    :eta, :zeta, :airyai, :airyaiprime, :airybi, :airybiprime,
    :airyaix, :airybix, :besselj0, :besselj1, :bessely0, :bessely1,
    :gamma, :lgamma, :loggamma, :logabsgamma, :lfact ]

    for func in ListOfSingleArgSpecialFuncs
       expr = :( SpecialFunctions.$func(x::DFP{N}) where {N} = call($func,x) )
       eval(expr)
    end

    ListOfDoubleArgSpecialFuncs = [ :besselj, :besseljx, :bessely,
    :besselyx, :hankelh1, :hankelh1x, :hankelh2, :hankelh2x,
    :besselhx, :besseli, :besselix, :besselk, :besselkx, :beta,
    :lbeta ]

    for func in ListOfDoubleArgSpecialFuncs
       expr = :( SpecialFunctions.$func(y::DFP{N},x::DFP{N}) where {N} = call($func,y,x) )
       eval(expr)
    end

    for func in [ :besselh ]
       expr = :( SpecialFunctions.$func(z::DFP{N},y::DFP{N},x::DFP{N}) where {N} = call($func,z,y,x) )
       eval(expr)
    end

    # Rounding
    round(x::DFP,r::RoundingMode; digits::Integer=0, sigdigits::Integer=0, base = 10) = DFP_round(x,r,digits=digits,sigdigits=sigdigits,base=base)
    round(x::DFP,r::RoundingMode{:NearestTiesAway}; digits::Integer=0, sigdigits::Integer=0, base = 10) = DFP_round(x,r,digits=digits,sigdigits=sigdigits,base=base)
    round(x::DFP,r::RoundingMode{:NearestTiesUp}; digits::Integer=0, sigdigits::Integer=0, base = 10) = DFP_round(x,r,digits=digits,sigdigits=sigdigits,base=base)


    # nextfloat, prevfloat and eps
    nextfloat(x::DFP) = x.s == 0 ? DFP_roundup(x) : DFP_rounddown(x)
    prevfloat(x::DFP) = x.s == 0 ? DFP_rounddown(x) : DFP_roundup(x)
    eps(x::DFP{N}) where {N} = max(x - prevfloat(x), nextfloat(x) - x)
    eps(::Type{DFP{N}}) where {N} = nextfloat(one(DFP{N})) - one(DFP{N})
    eps(::Type{DFP}) = nextfloat(one(DFP{DEFAULT_PRECISION[][Base.Threads.threadid()]})) - one(DFP{DEFAULT_PRECISION[][Base.Threads.threadid()]})

    # obtain the relative Epsilon
    function DFP_relativeEpsilon(a::DFP)
        len = Int64(length(a.m))
        return DFP{len}(0,a.expo - len + 1,vcat([1],DFP_listbuilder( len - 1)) )
    end

    function nextfloat(x::DFP,n::Integer)
        result = deepcopy(x)
        if n >= 0
            for count = 1:n
                result = nextfloat(result)
            end
        else
            for count = 1:(-1*n)
                result = prevfloat(result)
            end
        end
        return result
    end

    function prevfloat(x::DFP,n::Integer)
        result = deepcopy(x)
        if n >= 0
            for count = 1:n
                result = prevfloat(result)
            end
        else
            for count = 1:(-1*n)
                result = nextfloat(result)
            end
        end
        return result
    end

    # Dealing with Ipf Type
    @inline function MantissaStringInScientificFormat(a::DFP)
        s = string(FromDigits(a.m))
        s[1:1] * "."  * s[2:end]
    end

    function MantissaStringInLog10Format(a::DFP)
        if DFP_isZero(a)
            if DFP_lengthMantissa(a) > 1
                return "0." * string(FromDigits(a.m))[1:end-1]
            else
                return "0"
            end
        end
        if a.expo >= 0
            throw(DomainError(-1,"Error! a.expo must be negative in MantissaStringInLog10Format"))
        end
        s = string(FromDigits(a.m))
        n = abs(a.expo)-1
        "0." * "0"^n * s
    end

    @inline function takelog10ToIpfType(a::DFP)
        if a.s == 1
            throw(DomainError(-1,"Cannot take the log10 of a negative number in takelog10ToIpfType"))
        end
        prec = DFP_lengthMantissa(a)
        newprec = prec + GUARD_DIGITS
        numofbits = Int64(  ceil(3.322 * newprec)  )
        v = setprecision(numofbits) do
                log10(  parse(BigFloat,MantissaStringInScientificFormat(a))  )
            end
        IpfType(a.expo,DFP(v,prec))
    end

    @inline function straightToIpfType(a::DFP)
        frac = DFP_fraction(a)
        expo = parse(BigInt,DFP_toIntegerString(a))
        IpfType(expo,frac)
    end

    @inline function straightToIpfType(a::Integer)
        frac = DFP_createZero(DFP_getDefaultPrecision())
        IpfType(a,frac)
    end

    @inline function straightToIpfType(a::Integer,prec::Int64)
        frac = DFP_createZero(prec)
        IpfType(a,frac)
    end

    function +(a::IpfType,b::IpfType)::IpfType
        c = a.d + b.d
        if c.expo < 0 || DFP_isZero(c)
            return IpfType(a.int + b.int,c)
        end
        if c.expo == 0
            newexpo = a.int + b.int + c.m[1]
            c = c << 1
            return IpfType(newexpo,c)
        end
    end

    function -(a::IpfType,b::IpfType)::IpfType
        c = a.d - b.d
        if c.s == 1
            prec = DFP_lengthMantissa(c)
            one = DFP_createOne(prec)
            newc = one + c
            return IpfType(a.int - b.int - 1,newc)
        else
            return IpfType(a.int - b.int,c)
        end
    end

    function *(a::IpfType,b::IpfType)::IpfType
        # Stage 1
        stage1_int = a.int * b.int

        # Stage 2
        stringa = string(a.int)
        aintprec = length(stringa)
        bfracprec = DFP_lengthMantissa(b.d)
        newprec = max(aintprec,bfracprec) + GUARD_DIGITS
        aa = DFP(stringa,newprec)
        bb = DFP_setPrecision(b.d,newprec)
        rr = aa * bb
        intrr = DFP_integer(rr)
        stage2_int = parse(BigInt,DFP_toIntegerString(intrr))
        stage2_frac = DFP_setPrecision(DFP_fraction(rr),bfracprec)

        # Stage 3
        afracprec = DFP_lengthMantissa(a.d)
        stringb = string(b.int)
        bintprec = length(stringb)
        newprec = max(afracprec,bintprec) + GUARD_DIGITS
        aa = DFP_setPrecision(a.d,newprec)
        bb = DFP(stringb,newprec)
        rr = aa * bb
        intrr = DFP_integer(rr)
        stage3_int = parse(BigInt,DFP_toIntegerString(intrr))
        stage3_frac = DFP_setPrecision(DFP_fraction(rr),afracprec)

        # Stage 4
        afracprec = DFP_lengthMantissa(a.d)
        bfracprec = DFP_lengthMantissa(b.d)
        newprec = max(afracprec,bfracprec) + GUARD_DIGITS
        aa = DFP_setPrecision(a.d,newprec)
        bb = DFP_setPrecision(b.d,newprec)
        rr = aa * bb
        intrr = DFP_integer(rr)
        stage4_int = parse(BigInt,DFP_toIntegerString(intrr))
        stage4_frac = DFP_setPrecision(DFP_fraction(rr),afracprec)

        # Final Stage
        r = stage2_frac + stage3_frac + stage4_frac
        intr = DFP_integer(r)
        finalfrac = DFP_fraction(r)
        finalint = stage1_int + stage2_int +
                   stage3_int + stage4_int +
                   parse(BigInt,DFP_toIntegerString(intr))
        return IpfType(finalint,finalfrac)
    end

    function /(a::IpfType,b::IpfType)::IpfType
        prec = DFP_lengthMantissa(a.d)
        a_int_prec = length(string(a.int))
        b_int_prec = length(string(b.int))
        newprec = max(a_int_prec + 2 * (prec + GUARD_DIGITS), b_int_prec + 2 * (prec + GUARD_DIGITS))
        newa = DFP(a.int,newprec) + DFP(a.d,newprec)
        newb = DFP(b.int,newprec) + DFP(b.d,newprec)
        newc = newa / newb
        frac = DFP_setPrecision(DFP_fraction(newc),prec)
        expo = parse(BigInt,DFP_toIntegerString(newc))
        IpfType(expo,frac)
    end

    @inline function takeexp10OfIpfType(a::IpfType)
        prec = DFP_lengthMantissa(a.d)
        newprec = prec + GUARD_DIGITS
        numofbits = Int64(  ceil(3.322 * newprec)  )
        r = setprecision(numofbits) do
                exp10(  parse(BigFloat,MantissaStringInLog10Format(a.d))  )
            end
        temp = DFP(r,newprec)
        DFP_setPrecision(DFP{newprec}(0,a.int,temp.m),prec)
    end

    function DFP_power(a::DFP,b::Integer)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if iszero(b)
            return DFP_createOne(prec)
        end
        if DFP_isZero(a)
            return DFP_createZero(prec)
        end
        if DFP_isOne(a)
            return a
        end
        newprec = prec + GUARD_DIGITS
        if DFP_isPositive(a)
            aa = DFP_setPrecision(a,newprec)
            ipfaa = takelog10ToIpfType(aa)
            ipfbb = straightToIpfType(b,newprec)
            result = ipfbb * ipfaa
            # Handle the special case when the logarithmn
            # is negative by taking the fractional part
            # and add one to it and take the integer part
            # and subtract one from it
            if result.d.s == 1
                f = result.d
                len = DFP_lengthMantissa(f)
                newf = DFP_createOne(len) + f
                newint = result.int - 1
                result = IpfType(newint,newf)
            end
            return DFP_setPrecision( takeexp10OfIpfType( result ),prec )
        else
            # Since it is not Zero then it must be negative
            positive_a = DFP{prec}(0,a.expo,a.m)
            result = DFP_power(positive_a,b)
            if isodd(b)
                # If the power is an odd integer then the
                # result is negative
                result = DFP{prec}(1,result.expo,result.m)
            end
            return result
        end
    end

    function DFP_power(a::DFP{N},b::DFP{N}) where {N}
        # Check if a or b is an Error
        if DFP_isError(a)
            return a
        end
        if DFP_isError(b)
            return b
        end
        if DFP_isZero(b)
            return DFP_createOne(N)
        end
        if DFP_isZero(a)
            return DFP_createZero(N)
        end
        if DFP_isOne(a)
            return a
        end
        if DFP_isPositive(a)
            newprec = N + GUARD_DIGITS
            aa = DFP_setPrecision(a,newprec)
            bb = DFP_setPrecision(b,newprec)
            ipfaa = takelog10ToIpfType(aa)
            ipfbb = straightToIpfType(bb)
            result = ipfbb * ipfaa
            # Handle the special case when the logarithmn
            # is negative by taking the fractional part
            # and add one to it and take the integer part
            # and subtract one from it
            if result.d.s == 1
                f = result.d
                len = DFP_lengthMantissa(f)
                newf = DFP_createOne(len) + f
                newint = result.int - 1
                result = IpfType(newint,newf)
            end
            return DFP_setPrecision( takeexp10OfIpfType( result ),N )
        else
            # Since it is not Zero then a must be negative
            if DFP_isInteger(b)
                positive_a = DFP{N}(0,a.expo,a.m)
                result = DFP_power(positive_a,b)
                unitdigit = DFP_getUnitDigit(b)
                if unitdigit != nothing && isodd(unitdigit)
                    # If the power is an odd integer then the
                    # result is negative
                    result = DFP{N}(1,result.expo,result.m)
                end
                return result
            else
                # b is not an integer so the result is complex
                # return a domain error as we only deal in the Real domain
                return DFP_createError(N,3)
            end
        end # if DFP_isPositive(a)
    end

    # Calculate the ath root of b
    function DFP_nroot(a::DFP{N},b::DFP{N}) where {N}
        # Check if a or b is an Error
        if DFP_isError(a)
            return a
        end
        if DFP_isError(b)
            return b
        end
        if DFP_isZero(b) && ! DFP_isZero(a)
            return DFP_createZero(N)
        end
        if DFP_isZero(a)
            if  DFP_isOne(b)
                return DFP_createError(N,5)
            else
                return DFP_createError(N,4)
            end
        end
        if DFP_isOne(a)
            return b
        end
        if DFP_isNegative(b)
            # We cannot take the nroot of a negative number
            # return a domain error as we only deal in the Real domain
            return DFP_createError(N,3)
        end
        if DFP_isPositive(a)
            if a < DFP_createOne(N)
                return DFP_power(b,inv(a))
            else
                # a must be greater than one
                newprec = N + GUARD_DIGITS
                aa = DFP_setPrecision(a,newprec)
                bb = DFP_setPrecision(b,newprec)
                ipfbb = takelog10ToIpfType(bb)
                # we have to be very careful here
                # Stage 1 : Divide the integer part of ipfbb by aa
                temp_prec = newprec + length(string(ipfbb.int))
                integervalue = DFP(ipfbb.int,temp_prec)
                divisor = DFP_setPrecision(a,temp_prec)
                Stage1_result = integervalue / divisor
                Stage1_result_int = DFP_integer(Stage1_result)
                Stage1_result_frac = DFP_fraction(Stage1_result)
                Stage1_result_frac_newprec = DFP_setPrecision(Stage1_result_frac,newprec)
                # Stage 2 : Divide the fraction part of ipfbb by aa
                fractionvalue = ipfbb.d
                Stage2_result_frac = fractionvalue / aa
                # Final Stage : we need to combine the two result
                Final_int = parse(BigInt,DFP_toIntegerString(Stage1_result_int))
                Final_r = Stage1_result_frac_newprec + Stage2_result_frac
                intr = DFP_integer(Final_r)
                Final_frac = DFP_fraction(Final_r)
                Final_int += parse(BigInt,DFP_toIntegerString(intr))
                Final_Ipf = IpfType(Final_int,Final_frac)
                Final_DFP = DFP_setPrecision(  takeexp10OfIpfType(Final_Ipf), N  )
                return Final_DFP
            end
        else
            # Since it is not Zero then a must be negative
            # since a is negative the result must be complex
            # return a domain error as we only deal in the Real domain
            return DFP_createError(N,3)
        end # if DFP_isPositive(a)
    end # function DFP_nroot(a::DFP,b::DFP)

    nroot(a::DFP{N},b::DFP{N}) where {N} = DFP_nroot(a,b)

    nroot(a::Real,b::DFP{N}) where {N} = DFP_nroot(DFP{N}(a),b)

## Now we do all the mathematical constants

    # Handle the constant pi    \pi<tab>
    Base.convert(::Type{DFP{N}}, ::Irrational{:π}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.pi))
            end
        , N)

    DFP{N}(::Irrational{:π}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.pi))
            end
        , N)

    DFP(::Irrational{:π}) = DFP(
       setprecision(Int64(ceil(3.322*(DEFAULT_PRECISION[][Base.Threads.threadid()]+GUARD_DIGITS)))) do
           string(BigFloat(Base.MathConstants.pi))
       end
       ,DEFAULT_PRECISION[][Base.Threads.threadid()])

    DFP(::Irrational{:π},N::Int64) =
       DFP(
           setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
               string(BigFloat(Base.MathConstants.pi))
           end
       , N)

    # Handle the constant e    \euler<tab>

    Base.convert(::Type{DFP{N}}, ::Irrational{:ℯ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.e))
            end
        , N)

    DFP{N}(::Irrational{:ℯ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.e))
            end
        , N)

    DFP(::Irrational{:ℯ}) = DFP(
       setprecision(Int64(ceil(3.322*(DEFAULT_PRECISION[][Base.Threads.threadid()]+GUARD_DIGITS)))) do
           string(BigFloat(Base.MathConstants.e))
       end
       ,DEFAULT_PRECISION[][Base.Threads.threadid()])

    DFP(::Irrational{:ℯ},N::Int64) =
       DFP(
           setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
               string(BigFloat(Base.MathConstants.e))
           end
        , N)

    # Handle the constant catalan

    Base.convert(::Type{DFP{N}}, ::Irrational{:catalan}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.catalan))
            end
        , N)

    DFP{N}(::Irrational{:catalan}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.catalan))
            end
        , N)

    DFP(::Irrational{:catalan}) = DFP(
        setprecision(Int64(ceil(3.322*(DEFAULT_PRECISION[][Base.Threads.threadid()]+GUARD_DIGITS)))) do
            string(BigFloat(Base.MathConstants.catalan))
        end
    ,DEFAULT_PRECISION[][Base.Threads.threadid()])

    DFP(::Irrational{:catalan},N::Int64) =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.catalan))
            end
        , N)

    # Handle the constant eulergamma    \gamma<tab>

    Base.convert(::Type{DFP{N}}, ::Irrational{:γ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.eulergamma))
            end
        , N)

    DFP{N}(::Irrational{:γ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.eulergamma))
            end
        , N)

    DFP(::Irrational{:γ}) = DFP(
        setprecision(Int64(ceil(3.322*(DEFAULT_PRECISION[][Base.Threads.threadid()]+GUARD_DIGITS)))) do
            string(BigFloat(Base.MathConstants.eulergamma))
        end
    ,DEFAULT_PRECISION[][Base.Threads.threadid()])

    DFP(::Irrational{:γ},N::Int64) =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.eulergamma))
            end
        , N)

    # Handle the constant golden    \varphi<tab>

    Base.convert(::Type{DFP{N}}, ::Irrational{:φ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.golden))
            end
        , N)

    DFP{N}(::Irrational{:φ}) where {N} =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.golden))
            end
        , N)

    DFP(::Irrational{:φ}) = DFP(
        setprecision(Int64(ceil(3.322*(DEFAULT_PRECISION[][Base.Threads.threadid()]+GUARD_DIGITS)))) do
            string(BigFloat(Base.MathConstants.golden))
        end
    ,DEFAULT_PRECISION[][Base.Threads.threadid()])

    DFP(::Irrational{:φ},N::Int64) =
        DFP(
            setprecision(Int64(ceil(3.322 * N + GUARD_DIGITS))) do
                string(BigFloat(Base.MathConstants.golden))
            end
        , N)

    # Emulations of Mathematica functions
    @inline function FromDigits(list::Array{Int8,1})
        parse(BigInt,  join(map(z->Char(z+48),list))  )
    end

    @inline function IntegerDigits(n::Integer)
        [ Int8(digit) - Int8('0') for digit in string(n)]
    end

    function call(func::Function,a::DFP{N}) where {N}
        newprec = N + GUARD_DIGITS
        numofbits = Int64(  ceil(3.322 * newprec)  )
        return DFP(
            setprecision(numofbits) do
                func(  parse(BigFloat,string(a))  )
            end
        ,N)
    end

    function call(func::Function,a::Array{DFP{N}}) where {N}
        newprec = N + GUARD_DIGITS
        numofbits = Int64(  ceil(3.322 * newprec)  )
        result = func(
                     setprecision(numofbits) do
                         map(x->parse(BigFloat,string(x)),a)
                     end
                 )
        return map(x->DFP(x,N),result)
    end

    function call(func::Function,a::DFP{N},b::DFP{N}) where {N}
        newprec = N + GUARD_DIGITS
        numofbits = Int64(  ceil(3.322 * newprec)  )
        return DFP(
            setprecision(numofbits) do
                func(
                    parse(BigFloat,string(a)),
                    parse(BigFloat,string(b))
                )
            end
        ,N)
    end

    function call_viaFloat64(func::Function,a::DFP{N}) where {N}
        return DFP(
                func(  parse(Float64,DFP_toFortranString(a))  )
        ,N)
    end

    function call_viaFloat64(func::Function,a::DFP{N},b::DFP{N}) where {N}
        return DFP(
                func(
                    parse(Float64,DFP_toFortranString(a)),
                    parse(Float64,DFP_toFortranString(b))
                )
        ,N)
    end


    # Start of DFP functions
    """
        DFP_getDefaultPrecision()

    Return the default precision for DFP of the local thread. Each
    thread has its own DEFAULT_PRECISION
    """
    function DFP_getDefaultPrecision()
        DEFAULT_PRECISION[][Base.Threads.threadid()]
    end

    """
        DFP_setDefaultPrecision(prec::Int)

    Sets the default precision for DFP for the local thread. Each
    thread has its own DEFAULT_PRECISION
    """
    function DFP_setDefaultPrecision(prec::Int)
        global DEFAULT_PRECISION[][Base.Threads.threadid()] = prec < 0 ? 0 : prec
    end

    """
        DFP_setAllDefaultPrecision(prec::Int)

    Sets the default precision for DFP for the ALL threads. Each
    thread has its own DEFAULT_PRECISION
    """
    function DFP_setAllDefaultPrecision(prec::Int)
        for k = 1:Base.Threads.nthreads()
            global DEFAULT_PRECISION[][k] = prec < 0 ? 0 : prec
        end
    end

    """
        DFP_getShowStyle()

    Return the current ShowStyle symbol for the local thread.
    """
    function DFP_getShowStyle()
        SHOW_STYLE[][Base.Threads.threadid()]
    end

    """
        DFP_setShowStyle(x::Symbol)

        Controls how Base.show displays a DFP object for the local thread.
        The valid values are
        1.  :default  or  :canon         for canonical string output
        2.  :normal   or  :string        for complete string output
        3.  :short    or  :shortstring   for short string output
        4.  :tiny     or  :tinystring    for tiny string output
        5.  <num>                        for <num> digits string output
    """
    function DFP_setShowStyle(x::IntegerOrSymbol)
        if  ( x == :default  ||  x == :canon       ||
              x == :normal   ||  x == :string      ||
              x == :short    ||  x == :shortstring ||
              x == :tiny     ||  x == :tinystring  ||
              (typeof(x) <: Integer && x > 0)  )
            global SHOW_STYLE[][Base.Threads.threadid()] = x
        else
            DFP_setShowStyle_print_error_message()
        end
    end

    # If called with x that is not a Symbol then print error message
    DFP_setShowStyle(x) = DFP_setShowStyle_print_error_message()

    # If called with no arguements then print error message
    DFP_setShowStyle() = DFP_setShowStyle_print_error_message()

    function DFP_setShowStyle_print_error_message()
        print(
        "Error! Invalid symbol or no argument given.\n",
        "Controls how Base.show displays a DFP object\n",
        "The valid values are\n",
        "1.  :default  or  :canon         for canonical string output\n",
        "2.  :normal   or  :string        for complete string output\n",
        "3.  :short    or  :shortstring   for short string output\n",
        "4.  :tiny     or  :tinystring    for tiny string output\n",
        "5.  <num>                        for <num> digits string output\n"
        )
    end

    """
        DFP_temporarily_setDefaultPrecision(func::Function,prec::Int)

    Allows the coder to write short form codes such as below

    DFP_temporarily_setDefaultPrecision(8) do
        # body of code using 8 digits default precision
        ...
    end
    """
    function DFP_temporarily_setDefaultPrecision(func::Function,prec::Int)
        olddefaultprec = DFP_getDefaultPrecision()
        DFP_setDefaultPrecision( prec )
        func()
        DFP_setDefaultPrecision( olddefaultprec )
    end

    @inline function DFP_listbuilder(n)
        zeros(Int8,n)
    end

    @inline function DFP_normalise(a::DFP{N}) where {N}
        # Do not normalise if it is an Error
        if DFP_isError(a)
            return a
        end
        expo = a.expo
        mantissa = deepcopy(a.m)
        if any(  map(x->x>0,mantissa)  )
            while mantissa[1] == 0
                popfirst!(mantissa)
                push!(mantissa,0)
                expo -= 1
            end
        else
            # It is a zero
            return DFP_createZero(length(mantissa))
        end
        return DFP{N}(a.s, expo, mantissa)
    end

    function Base.show(io::IO,x::DFP)
        if ( SHOW_STYLE[][Base.Threads.threadid()] == :default ||
             SHOW_STYLE[][Base.Threads.threadid()] == :canon )
            print(io,DFP_toCanonicalString(x))
        elseif ( SHOW_STYLE[][Base.Threads.threadid()] == :normal ||
                 SHOW_STYLE[][Base.Threads.threadid()] == :string )
            print(io,DFP_toCommonString(x))
        elseif ( SHOW_STYLE[][Base.Threads.threadid()] == :short ||
                 SHOW_STYLE[][Base.Threads.threadid()] == :shortstring )
            print(io,DFP_toShortCommonString(x))
        elseif ( SHOW_STYLE[][Base.Threads.threadid()] == :tiny ||
                 SHOW_STYLE[][Base.Threads.threadid()] == :tinystring )
            print(io,DFP_toTinyCommonString(x))
        elseif ( typeof(SHOW_STYLE[][Base.Threads.threadid()]) <: Integer )
            print(io,DFP_toNumCommonString(x,SHOW_STYLE[][Base.Threads.threadid()]))
        else
            print(io,"Error unknown show style ",SHOW_STYLE[][Base.Threads.threadid()]," detected.")
        end
    end

    function DFP_fromString(str::AbstractString)
        signvalue = Int8(0)
        expovalue = BigInt(0)
        mantissavalue = Array{Int8,1}[]
        decimalpointpos = 0
        expoadjustment = BigInt(0)
        regexSpecialNum = r"^ *(?<specialnum>(-Inf|Inf|NaN)) *$"
        m = match(regexSpecialNum,str)
        if m !== nothing
            if m[:specialnum] == "Inf"
                return DFP_createInfError(4)
            elseif m[:specialnum] == "-Inf"
                return DFP_createNegInfError(4)
            elseif m[:specialnum] == "NaN"
                return DFP_createNaNError(4)
            end
        end
        regexSciNum = r"^ *(?<sign>\+|-)?(?<mantissa>\d+(\.(\d+)?)?|\.\d+)((e|E)(?<expo>(\+|-)?\d+)?)? *$"
        m = match(regexSciNum,str)
        if m === nothing
            # It is posible that string is a valid string of Rational
            # so call the sub function DFP_fromRationalString
            local result = DFP_fromRationalString(str)
            if result == DFP{1}(0,1,[0])
                println("String <$str> does not match regexSciNum")
                return DFP{1}(0,1,[0])
            else
                return result
            end
        else
            # println("String $s match regexSciNum")
            if m[:sign] === nothing
                signstring = ""
            else
                signstring = m[:sign]
                if signstring == "-"
                    signvalue = Int8(1)
                end
            end
            if m[:expo] === nothing
                expostring = ""
            else
                expostring = m[:expo]
                expovalue = parse(BigInt,expostring)
            end
            if m[:mantissa] === nothing
                mantissastring = ""
            else
                mantissastring = m[:mantissa]
                mantissavalue = map(x -> parse(Int8,x),split(replace(mantissastring, "." => ""),""))
                decimalpointpos = findfirst(isequal('.'), mantissastring)
                if decimalpointpos === nothing
                    decimalpointpos = 0
                end
                if decimalpointpos > 0
                    expoadjustment += BigInt(decimalpointpos) - BigInt(2)
                else
                    expoadjustment += BigInt(length(mantissavalue)) - BigInt(1)
                end
            end
            expovalue += expoadjustment
            N = length(mantissavalue)
            return DFP_normalise(DFP{N}(signvalue,expovalue,mantissavalue))
        end
    end

    function DFP_fromRationalString(str::AbstractString)
        regexRationalNum = r"^ *(?<numerator>(\+|-)?\d+)//(?<denominator>(\+|-)?\d+) *$"
        m = match(regexRationalNum,str)
        if m === nothing
            println("String <$str> does not match regexRationalNum")
            return DFP{1}(0,1,[0])
        else
            # println("String $s match regexRationalNum")
            numeratorstring = m[:numerator]
            denominatorstring = m[:denominator]
            numerator = DFP_fromString(numeratorstring)
            denominator = DFP_fromString(denominatorstring)
            maxprec = max(DFP_lengthMantissa(numerator),DFP_lengthMantissa(denominator))
            numerator = DFP_setPrecision(numerator,maxprec)
            denominator = DFP_setPrecision(denominator,maxprec)
            result = numerator / denominator
            return result
        end
    end

    function DFP{N}(str::AbstractString) where {N}
        # remove lagging spaces
        newstr = replace(str,r"\s+$" => s"")
        # replace p or P with comma (,)
        newstr = replace(newstr,r"p|P"=>",")
        if occursin(',',newstr)
            a = split(newstr,",")
            DFP_setPrecision(DFP_fromString(a[1]),N)
        else
            DFP_setPrecision(DFP_fromString(str),N)
        end
    end

    function DFP(str::AbstractString)
        # remove lagging spaces
        newstr = replace(str,r"\s+$" => s"")
        # replace p or P with comma (,)
        newstr = replace(newstr,r"p|P"=>",")
        if occursin(',',newstr)
            a = split(newstr,",")
            prec = tryparse(Int64,a[2])
            if prec == nothing
                DFP_setPrecision(DFP_fromString(a[1]),DEFAULT_PRECISION[][Base.Threads.threadid()])
            else
                DFP_setPrecision(DFP_fromString(a[1]),prec)
            end
        else
            DFP_setPrecision(DFP_fromString(str),DEFAULT_PRECISION[][Base.Threads.threadid()])
        end
    end

    DFP(str::AbstractString, prec) = DFP_setPrecision(DFP_fromString(str),prec)



    @inline function DFP_lengthMantissa(a::DFP)
        return Int64(length(a.m))
    end

    function DFP_mapBinaryFuncToList(func::Function,list)
        a = deepcopy(list);
        while length(a)>1
            newa = Array{eltype(a),1}[]
            while length(a) > 1
                push!(newa,func( a[1],a[2] ))
                # Remove the first two items from a
                popfirst!(a)
                popfirst!(a)
            end
            if length(a) == 1
                push!(newa,a[1])
            end
            a = deepcopy(newa)
        end
        a[1]
    end

    function DFP_convertOtherFloatToCommonString(x)
        # Limit the convertion to only the first 100 decimal digits
        local result = string(convert(DFP{100},x))
        result = rstrip(x->x=='0',result)
        if last(result)=='.'
            result = result * '0'
        end
        return result
    end

    function DFP_toFortranString(a::DFP;EString::String="E")
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        if length(a.m) == 0
            # Handle edge case of empty mantissa
            return ""
        end
        stringlist=String[]
        if a.s > 0
            push!(stringlist,"-")
        end
        if length(a.m)>0
            push!(stringlist,string(a.m[1]))
        end
        temp = a.m[2:end]
        if length(temp)>0
            push!(stringlist,".")
            push!(stringlist,join(string.(temp))  )
        end
        #push!(stringlist,"E")  REPLACED BELOW WITH EString
        push!(stringlist,EString)
        push!(stringlist,string(a.expo))
        return join(stringlist)
    end

    function DFP_toCommonString(a::DFP;EString::String="E",MinExpo::Int64=4)
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        if length(a.m) == 0
            # Handle edge case of empty mantissa
            return ""
        end
        stringlist=String[]
        if a.s > 0
            push!(stringlist,"-")
        end
        # if the expo is negative and less than -6 then
        # call the DFP_toFortranString
        if a.expo < 0
            # expo is less than 0
            # The parameter MinExpo cannot be less than zero
            if a.expo < -(MinExpo < 0 ? 0 : MinExpo)
                return DFP_toFortranString(a; :EString => EString)
            end
            # -(MinExpo+1) < a.expo < 0
            numofzero = abs(a.expo) - 1
            push!(    stringlist,string("0.","0" ^ numofzero)    )
            push!(    stringlist,join(string.(a.m))    )
            return join(stringlist)
        else
            # expo is equal to or greater than 0
            len = DFP_lengthMantissa(a)
            if a.expo < len
                if a.expo == len - 1
                    # then we return the whole mantissa
                    push!(    stringlist,join(string.(a.m))    )
                else
                    # We need to split the mantissa into two parts
                    firstpart = a.m[1:a.expo+1]
                    secondpart = a.m[a.expo+2:end]
                    push!(    stringlist,join(string.(firstpart))    )
                    push!(    stringlist,"."    )
                    push!(    stringlist,join(string.(secondpart))    )
                end
                return join(stringlist)
            else
                # expo is greater than len then
                # call the DFP_toFortranString
                return DFP_toFortranString(a; :EString => EString)
            end
        end
    end

    function DFP_toIntegerString(a::DFP)
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        if length(a.m) == 0
            # Handle edge case of empty mantissa
            return ""
        end
        IntegerA = DFP_integer(a)
        stringlist=String[]
        prec = DFP_lengthMantissa(IntegerA)
        # Sanity Check
        if IntegerA.expo < 0
            return "0"
        end
        # If IntegerA is zero
        if DFP_isZero(IntegerA)
            return "0"
        end
        if IntegerA.s > 0
            push!(stringlist,"-")
        end
        arrayofchar = map(x->Char(x+48),IntegerA.m)
        if IntegerA.expo + 1 > prec
            push!(stringlist,  join(arrayofchar)  )
            push!(stringlist,  "0"^(IntegerA.expo + 1 - prec)  )
        else
            push!(stringlist,  join(arrayofchar[1:IntegerA.expo+1])  )
        end
        return join(stringlist)
    end

    function DFP_toShortCommonString(a::DFP)
        if DFP_lengthMantissa(a) > 6
            return DFP_toCommonString(  DFP_setPrecision(a,6)  )
        else
            return DFP_toCommonString(a)
        end
    end

    function DFP_toTinyCommonString(a::DFP)
        if DFP_lengthMantissa(a) > 4
            return DFP_toCommonString(  DFP_setPrecision(a,4)  , MinExpo=2  )
        else
            return DFP_toCommonString(a)
        end
    end

    function DFP_toNumCommonString(a::DFP,n::Integer)
        if DFP_lengthMantissa(a) > n
            return DFP_toCommonString(  DFP_setPrecision(a,n)  , MinExpo=2  )
        else
            return DFP_toCommonString(a)
        end
    end

    function DFP_toExplainationString(a::DFP)
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        stringlist=String[]
        if a.s > 0
            push!(stringlist,"-1 * ")
        else
            push!(stringlist,"+1 * ")
        end
        if length(a.m) == 0
            push!(stringlist,"anynumber")
        else
            push!(stringlist,string(a.m[1]))
        end
        temp = a.m[2:end]
        if length(temp)>0
            push!(stringlist,".")
            push!(stringlist,join(string.(temp))  )
        end
        push!(stringlist," * 10^")
        push!(stringlist,string(a.expo))
        return join(stringlist)
    end

    function DFP_toNdigitString(a::DFP,n::Int64)
        prec = DFP_lengthMantissa(a)
        if n > prec
            n = prec
        end
        if n == 0 || n < 0
            # if user ask for 0 digits then return empty string
            return ""
        end
        return DFP_toCommonString(  DFP_setPrecision(a,n)  )
    end

    function DFP_toCanonicalString(a::DFP{N}) where {N}
        stringlist=String[]
        push!(stringlist,"DFP{",string(N),"}(")
        push!(stringlist,string(a.s))
        push!(stringlist,", ")
        push!(stringlist,string(a.expo))
        push!(stringlist,", [")
        if length(a.m) > 0
            for k = 1:length(a.m)-1
                push!(stringlist,string(a.m[k]) * ", ")
            end
            push!(stringlist,string(a.m[end]))
        end
        push!(stringlist,"])")
        return join(stringlist)
    end

    function DFP_toMantissaString(a::DFP)
        return join(string.(a.m))
    end

    function DFP_toHardFloatingPointString(a::DFP;TenString::String="10.0")
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        if length(a.m) == 0
            # Handle edge case of empty mantissa
            return ""
        end
        stringlist=String[]
        if a.s > 0
            push!(stringlist,"-")
        end
        if length(a.m)>0
            push!(stringlist,string(a.m[1]))
        end
        temp = a.m[2:end]
        if length(temp)>0
            push!(stringlist,".")
            push!(stringlist,join(string.(temp))  )
        end
        push!(stringlist,"*" * TenString * "^")
        push!(stringlist,string(a.expo))
        return join(stringlist)
    end

    function DFP_toMathematicaString(a::DFP;TenString::String="10.0")
        # First check if it is an Error
        if DFP_isDivByZero(a)
            return "Error 0 : Division By Zero"
        end
        if DFP_isOtherError(a)
            key = Int64(a.expo)
            if haskey(ErrorMessage,key)
                return ErrorMessage[key]
            else
                return join(["Error ",string(key)," : unknown error message"])
            end
        end
        # No errors so we continue
        if length(a.m) == 0
            # Handle edge case of empty mantissa
            return ""
        end
        stringlist=String[]
        if a.s > 0
            push!(stringlist,"-")
        end
        # if the expo is negative and less than -6 then
        # call the DFP_toFortranString
        if a.expo < 0
            # expo is less than 0
            if a.expo < -4
                return DFP_toHardFloatingPointString(a; :TenString => TenString)
            end
            # -5 < a.expo < 0
            numofzero = abs(a.expo) - 1
            push!(    stringlist,string("0.","0" ^ numofzero)    )
            push!(    stringlist,join(string.(a.m))    )
            return join(stringlist)
        else
            # expo is equal to or greater than 0
            len = DFP_lengthMantissa(a)
            if a.expo < len
                if a.expo == len - 1
                    # then we return the whole mantissa
                    push!(    stringlist,join(string.(a.m))    )
                else
                    # We need to split the mantissa into two parts
                    firstpart = a.m[1:a.expo+1]
                    secondpart = a.m[a.expo+2:end]
                    push!(    stringlist,join(string.(firstpart))    )
                    push!(    stringlist,"."    )
                    push!(    stringlist,join(string.(secondpart))    )
                end
                return join(stringlist)
            else
                # expo is greater than len then
                # call the DFP_toFortranString
                return DFP_toHardFloatingPointString(a; :TenString => TenString)
            end
        end
    end

    function DFP_toFloat64(x::DFP)
        prec = DFP_lengthMantissa(x)
        # Float64
        DBL_MIN_string = "2.22507385850720138e-308"
        DBL_MAX_string = "1.79769313486231571e+308"
        DBL_MIN = DFP(DBL_MIN_string,prec)
        DBL_MAX = DFP(DBL_MAX_string,prec)
        NEG_DBL_MIN_string = "-" * DBL_MIN_string
        NEG_DBL_MAX_string = "-" * DBL_MAX_string
        NEG_DBL_MIN = DFP(NEG_DBL_MIN_string,prec)
        NEG_DBL_MAX = DFP(NEG_DBL_MAX_string,prec)
        if DFP_isError(x)
            return Base.parse(Float64,"NaN")
        end
        if DFP_isZero(x)
            return Float64(0)
        end
        if x > DFP_0(prec)
            if x > DBL_MAX
                return Base.parse(Float64,"Inf")
            elseif x < DBL_MIN
                return Float64(0)
            else
                return Base.parse(Float64,string(x))
            end
        else
            if x < NEG_DBL_MAX
                return Base.parse(Float64,"-Inf")
            elseif x > NEG_DBL_MIN
                return Float64(0)
            else
                return Base.parse(Float64,string(x))
            end
        end
    end

    @inline function DFP_createZero(prec)
        return DFP{prec}(0,0,DFP_listbuilder(prec))
    end

    const DFP_0 = DFP_createZero

    @inline function DFP_createOne(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([1],DFP_listbuilder(prec - 1)))
    end

    const DFP_1 = DFP_createOne

    @inline function DFP_createHalf(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,-1,vcat([5],DFP_listbuilder(prec - 1)))
    end

    const DFP_half = DFP_createHalf

    @inline function DFP_createOneTenth(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,-1,vcat([1],DFP_listbuilder(prec - 1)))
    end

    @inline function DFP_createMinusOne(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(1,0,vcat([1],DFP_listbuilder(prec - 1)))
    end

    @inline function DFP_createTwo(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([2],DFP_listbuilder(prec - 1)))
    end

    const DFP_2 = DFP_createTwo

    @inline function DFP_createThree(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([3],DFP_listbuilder(prec - 1)))
    end

    const DFP_3 = DFP_createThree

    @inline function DFP_createFour(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([4],DFP_listbuilder(prec - 1)))
    end

    const DFP_4 = DFP_createFour

    @inline function DFP_createFive(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([5],DFP_listbuilder(prec - 1)))
    end

    const DFP_5 = DFP_createFive

    @inline function DFP_createEight(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,0,vcat([8],DFP_listbuilder(prec - 1)))
    end

    const DFP_8 = DFP_createEight

    @inline function DFP_createTen(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([1],DFP_listbuilder(prec - 1)))
    end

    const DFP_10 = DFP_createTen

    @inline function DFP_createFifteen(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([1,5],DFP_listbuilder(prec - 2)))
    end

    const DFP_15 = DFP_createFifteen

    @inline function DFP_createThirty(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([3],DFP_listbuilder(prec - 1)))
    end

    const DFP_30 = DFP_createThirty

    @inline function DFP_createFortyFive(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([4,5],DFP_listbuilder(prec - 2)))
    end

    const DFP_45 = DFP_createFortyFive

    @inline function DFP_createSixty(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([6],DFP_listbuilder(prec - 1)))
    end

    const DFP_60 = DFP_createSixty

    @inline function DFP_createSixtyFour(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([6,4],DFP_listbuilder(prec - 2)))
    end

    const DFP_64 = DFP_createSixtyFour

    @inline function DFP_createNinety(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,1,vcat([9],DFP_listbuilder(prec - 1)))
    end

    const DFP_90 = DFP_createNinety

    @inline function DFP_createOneEighty(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,2,vcat([1,8],DFP_listbuilder(prec - 2)))
    end

    const DFP_180 = DFP_createOneEighty

    @inline function DFP_createTwoSeventy(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,2,vcat([2,7],DFP_listbuilder(prec - 2)))
    end

    const DFP_270 = DFP_createTwoSeventy

    @inline function DFP_createThreeSixty(prec)
        return prec == 0 ? DFP{0}(0,0,[]) : DFP{prec}(0,2,vcat([3,6],DFP_listbuilder(prec - 2)))
    end

    const DFP_360 = DFP_createThreeSixty

    @inline function DFP_createDivByZeroError(prec)
        return DFP{prec}(1,0,DFP_listbuilder(prec))
    end

    @inline function DFP_createError(prec,n)
        return DFP{prec}(0,n,DFP_listbuilder(prec))
    end

    @inline function DFP_createInfError(prec)
        return DFP{prec}(0,7,DFP_listbuilder(prec))
    end

    @inline function DFP_createNegInfError(prec)
        return DFP{prec}(0,8,DFP_listbuilder(prec))
    end

    @inline function DFP_createNaNError(prec)
        return DFP{prec}(0,9,DFP_listbuilder(prec))
    end

    @inline function DFP_isZero(a::DFP)
        if a.s == 0 && a.expo == 0 && all(  map(x->x==0,a.m)  )
            return true
        else
            return false
        end
    end

    @inline function DFP_isOne(a::DFP)
        if a.s == 0 && a.expo == 0 && all( map(x->x==0,a.m[2:end]) ) && a.m[1] == 1
            return true
        else
            return false
        end
    end

    @inline function DFP_isMinusOne(a::DFP)
        if a.s == 1 && a.expo == 0 && all( map(x->x==0,a.m[2:end]) ) && a.m[1] == 1
            return true
        else
            return false
        end
    end

    @inline function DFP_isDivByZero(a::DFP)
        if a.s == 1 && a.expo == 0 && all( map(x->x==0,a.m) )
            return true
        else
            return false
        end
    end

    @inline function DFP_isOtherError(a::DFP)
        # Any Error other than DivByZero
        if a.s == 0 && a.expo > 0 && all( map(x->x==0,a.m) )
            return true
        else
            return false
        end
    end

    @inline function DFP_isError(a::DFP)
        if DFP_isDivByZero(a) || DFP_isOtherError(a)
            return true
        else
            return false
        end
    end

    @inline function DFP_isNormalNumber(a::DFP)
        return ! DFP_isError(a)
    end

    @inline function DFP_isSpecialNumber(a::DFP)
        if a.s == 0 && (7 <= a.expo <= 9) && all( map(x->x==0,a.m) )
            return true
        else
            return false
        end
    end

    @inline function DFP_isOrdinaryError(a::DFP)
        if DFP_isNormalNumber(a) || DFP_isSpecialNumber(a)
            return false
        else
            return DFP_isError(a)
        end
    end

    @inline DFP_isNaN(a::DFP) = DFP_kind(a) == 3

    @inline DFP_isInf(a::DFP) = DFP_kind(a) == 1

    @inline DFP_isNegInf(a::DFP) = DFP_kind(a) == 2

    """
        DFP_kind(a::DFP)

    returns the kind of DFP
        0    Normal Number
        1    Inf
        2    -Inf
        3    NaN
        4    Ordinary Error
    """
    @inline function DFP_kind(a::DFP)
        # First check if mantissa is all zero
        if all(  map(x->x==0,a.m)  )
            # All the mantissa digits are zero
            if a.s == 0
                if a.expo == 0
                    # The number is zero   (signbit==0 and expo==0)
                    return 0
                elseif (0 < a.expo < 7) || (9 < a.expo)
                    # Ordinary Error
                    return 4
                elseif a.expo == 7
                    # Inf          (signbit==0 and expo==7)
                    return 1
                elseif a.expo == 8
                    # -Inf         (signbit==0 and expo==8)
                    return 2
                elseif a.expo == 9
                    # NaN          (signbit==0 and expo==9)
                    return 3
                end
            else   # a.s != 0      signbit is not 0
                if a.expo == 0
                    # Critical Error
                    return 4
                end
            end
        else
            # Not all mantissa is zero so it is a Normal Number
            return 0
        end
    end

    @inline DFP_lhsrhs_kind(lhs::DFP,rhs::DFP) = (DFP_kind(lhs),DFP_kind(rhs))

    @inline function DFP_isPositive(a::DFP)
        if DFP_isError(a) == false && DFP_isZero(a) == false && a.s == 0
            return true
        else
            return false
        end
    end

    @inline function DFP_isNegative(a::DFP)
        if DFP_isError(a) == false && DFP_isZero(a) == false && a.s == 1
            return true
        else
            return false
        end
    end

    @inline function DFP_isEqual(a::DFP,b::DFP)
        return a.s == b.s && a.expo == b.expo && a.m == b.m
    end

    @inline function DFP_isInteger(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return false
        end
        temp = DFP_integer(a)
        return DFP_isEqual(temp,a)
    end

    @inline function DFP_isFraction(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return false
        end
        temp = DFP_fraction(a)
        return DFP_isEqual(temp,a)
    end

    """
        DFP_round(x::DFP,r::RoundingMode=RoundNearest; digits::Integer=0,
                   sigdigits::Integer=0, base = 10)

        Provides rounding facilities for DFP. Uses significant digits if
        sigdigits > 0 otherwise uses digits

        The RoundingModes are RoundNearest(default),RoundNearestTiesAway,
        RoundNearestTiesUp, RoundToZero, RoundUp and RoundDown.

        Calls to Base.round(::DFP,RoundingMode) are mapped to here.

        RoundNearest  (default)
        Rounds to the nearest integer, with ties (fractional values of 0.5) being rounded to the nearest even integer.

        RoundNearestTiesAway
        Rounds to nearest integer, with ties rounded away from zero

        RoundNearestTiesUp
        Rounds to nearest integer, with ties rounded toward positive infinity

        RoundToZero
        Round towards zero

        RoundUp
        Round towards positive infinity

        RoundDown
        Round towards negative infinity
    """
    function DFP_round(x::DFP,r::RoundingMode=RoundNearest; digits::Integer=0, sigdigits::Integer=0, base = 10)
        prec = DFP_lengthMantissa(x)
        if sigdigits > 0
            "Adjust using significant digits"
            expodiff = (sigdigits-1) - x.expo
        else
            "Adjust using digits"
            expodiff = digits
        end
        " Step 1. perform the adjustment to the abs value of x "
        adjustedx = DFP{prec}(0,x.expo+expodiff,x.m)
        if r == RoundNearest
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to nearest "
                tempx = DFP_setPrecision(adjustedx,Int64(adjustedx.expo)+1)
                tempx = DFP_setPrecision(tempx,prec)
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            elseif adjustedx.expo == -1
                " Special case if expo is minus one "
                newprec = prec+2
                superx = DFP_setPrecision(adjustedx,newprec) + DFP_createTen(newprec)
                " Step 3. perform rounding to nearest "
                tempx = DFP_setPrecision(superx,2)
                tempx = DFP_setPrecision(tempx,newprec) - DFP_createTen(newprec)
                tempx = DFP_setPrecision(tempx,prec)
                " Step 4. undo adjustment and restore sign "
                if DFP_isZero(tempx)
                    return tempx
                end
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                " Special case if expo is less than minus one "
                return DFP_createZero(prec)
            end
        elseif r == RoundNearestTiesAway
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to RoundNearestTiesAway "
                tempintx = DFP_integer(adjustedx)
                tempfracx = DFP_fraction(adjustedx)
                if tempfracx >= DFP_createHalf(prec)
                    tempx = tempintx + DFP_createOne(prec)
                else
                    tempx = tempintx
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            elseif adjustedx.expo == -1
                " Special case if expo is minus one "
                " Step 3. perform rounding to nearest "
                if adjustedx >= DFP_createHalf(prec)
                    tempx = DFP_createOne(prec)
                else
                    return DFP_createZero(prec)
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                " Special case if expo is less than minus one "
                return DFP_createZero(prec)
            end
        elseif r == RoundNearestTiesUp
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to RoundNearestTiesUp "
                tempintx = DFP_integer(adjustedx)
                tempfracx = DFP_fraction(adjustedx)
                if x.s == 0
                    if tempfracx >= DFP_createHalf(prec)
                        tempx = tempintx + DFP_createOne(prec)
                    else
                        tempx = tempintx
                    end
                else # x.s != 0
                    if tempfracx <= DFP_createHalf(prec)
                        tempx = tempintx
                    else
                        tempx = tempintx + DFP_createOne(prec)
                    end
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            elseif adjustedx.expo == -1
                " Special case if expo is minus one "
                " Step 3. perform rounding to nearest "
                if x.s == 0
                    if adjustedx >= DFP_createHalf(prec)
                        tempx = DFP_createOne(prec)
                    else
                        return DFP_createZero(prec)
                    end
                else
                    if adjustedx <= DFP_createHalf(prec)
                        return DFP_createZero(prec)
                    else
                        tempx = DFP_createOne(prec)
                    end
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                " Special case if expo is less than minus one "
                return DFP_createZero(prec)
            end
        elseif r == RoundToZero
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to Zero "
                tempx = DFP_integer(adjustedx)
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                return DFP_createZero(prec)
            end
        elseif r == RoundUp
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to RoundUp "
                tempx = DFP_integer(adjustedx)
                if x.s == 0 && DFP_fraction(adjustedx) > DFP_createZero(prec)
                    tempx += DFP_createOne(prec)
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            elseif adjustedx.expo == -1
                " Special case if expo is minus one "
                " Step 3. perform rounding to RoundUp "
                if x.s == 0
                    tempx = DFP_createOne(prec)
                else
                    return DFP_createZero(prec)
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                " Special case if expo is less than minus one "
                if x.s == 0
                    tempx = DFP_createOne(prec)
                    return DFP{prec}(0,tempx.expo-expodiff,tempx.m)
                else
                    return DFP_createZero(prec)
                end
            end
        elseif r == RoundDown
            " Step 2. check if the expo is more than 0 "
            if adjustedx.expo > -1
                " Step 3. perform rounding to RoundDown "
                tempx = DFP_integer(adjustedx)
                if x.s == 1 && DFP_fraction(adjustedx) > DFP_createZero(prec)
                    tempx += DFP_createOne(prec)
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            elseif adjustedx.expo == -1
                " Special case if expo is minus one "
                " Step 3. perform rounding to RoundDown "
                if x.s == 0
                    return DFP_createZero(prec)
                else
                    tempx = DFP_createOne(prec)
                end
                " Step 4. undo adjustment and restore sign "
                return DFP{prec}(x.s,tempx.expo-expodiff,tempx.m)
            else
                " Special case if expo is less than minus one "
                if x.s == 0
                    return DFP_createZero(prec)
                else
                    tempx = DFP_createOne(prec)
                    return DFP{prec}(1,tempx.expo-expodiff,tempx.m)
                end
            end
        end
    end

    # Warning! DFP_roundup does not care about the sign of the DFP
    # therefore it always roundup away from zero
    @inline function DFP_roundup(a::DFP)
        len = length(a.m)
        if DFP_isZero(a)
            local one = DFP_createOne(len)
            return DFP_roundup(one) - one
        end
        rlist = IntegerDigits(  FromDigits(a.m) + BigInt(1)  )
        if length(rlist) > len
            return DFP{len}(a.s, a.expo + 1, rlist[1:len])
        else
            return DFP{len}(a.s, a.expo, rlist)
        end
    end

    # Warning! DFP_rounddown does not care about the sign of the DFP
    # therefore it always rounddown towards zero
    @inline function DFP_rounddown(a::DFP)
        len = length(a.m)
        if DFP_isZero(a)
            local one = DFP_createOne(len)
            return DFP_rounddown(one) - one
        end
        rlist = IntegerDigits(  FromDigits(a.m) - BigInt(1)  )
        if length(rlist) < len
            # Redo the subtraction again with extra zero
            rlist = IntegerDigits(  FromDigits(a.m) * BigInt(10) - BigInt(1)  )
            return DFP{len}(a.s, a.expo - 1, rlist[1:len])
        else
            return DFP{len}(a.s, a.expo, rlist)
        end
    end

    @inline function DFP_getPrecision(a::DFP)
        return DFP_lengthMantissa(a)
    end

    function DFP_setPrecision(a::DFP,desiredprec::Int64)
        # Handle special case when desiredprec is less than zero
        if desiredprec < 0
            return DFP_setPrecision(a,0)
        end
        # Handle special cases when the desiredprec is 0
        if desiredprec == 0
            if DFP_isError(a)
                # Return back the same Error state with empty mantissa array
                return DFP{0}(a.s,a.expo,[])
            else
                # In a zero precision Decimal Floating Point System
                # all (non-Error) real numbers have the same representation
                # DFP(0,0,[])
                return DFP{0}(0,0,[])
            end
        end
        # Now handle all the normal cases
        len = length(a.m)
        # If it is exactly the precision we wanted
        if len == desiredprec
            return a
        end
        # If it is less than the precision we wanted
        if len < desiredprec
            return DFP{desiredprec}(  a.s, a.expo, vcat(a.m,DFP_listbuilder(desiredprec - len))  )
        end
        # It is more than the precision we wanted
        lastdigit = a.m[desiredprec+1]
        result = DFP{desiredprec}(a.s,a.expo,a.m[1:desiredprec])
        if lastdigit > 5
            return DFP_roundup(result)
        end
        if lastdigit == 5 && isodd(result.m[desiredprec])
            return DFP_roundup(result)
        end
        if lastdigit == 5
            if any(  map(x->x>0,a.m[desiredprec+2:end])  )
                return DFP_roundup(result)
            else
                return result
            end
        end
        # The lastdigit is less than 5
        return result
    end

    # Function oda : One Digit Adder. The most primitive of adder.
    # add "a" with "b" with carry "c" then output result "r" with carry "d"
    function oda(a,b,c)
        d = 0
        r = a + b + c
        if r > 9
            d = div(r,10)
            r = r - 10 * d
        end
        return [r,d]
    end

    # This is the Original mda
    # We write it here for historical purposes
    function mdaOrig(arrayA,arrayB)
        precA = length(arrayA)
        precB = length(arrayB)
        # Perform sanity checking
        if precA != precB
            return []
        end
        result = DFP_listbuilder(precA)
        r = 0
        carry = 0
        for k = precA:-1:1
            (r,carry) = oda(arrayA[k],arrayB[k],carry)
            result[k] = r
        end
        return (result,carry)
    end

    # Multi Digits Adder
    # The next level up from oda
    @inline function mda(arrayA::Array{Int8,1},arrayB::Array{Int8,1})
        precA = length(arrayA)
        precB = length(arrayB)
        # Perform sanity checking
        if precA != precB
            return []
        end
        result = DFP_listbuilder(precA)
        r = 0
        carry = 0
        @inbounds for k = precA:-1:1
            r = arrayA[k] + arrayB[k] + carry
            if r > 9
                carry = div(r,10)
                r = r - 10 * carry
            else
                carry = 0
            end
            result[k] = r
        end
        return (result,carry)
    end

    # If the function mda is called with arrays of Int64
    # then convert it back to arrays of Int8
    function mda(arrayA::Array{Int64,1},arrayB::Array{Int64,1})
        newarrayA = Int8.(arrayA)
        newarrayB = Int8.(arrayB)
        return mda(newarrayA,newarrayB)
    end

    @inline function ninecomplement(arrayA::Array{Int8,1})
        return (x -> Int8(9 - x)).(arrayA)
    end

    # Convert an Array of Int64 into an Array of Int8
    @inline function ninecomplement(arrayA::Array{Int64,1})
        return (x -> Int8(9 - x)).(arrayA)
    end

    @inline function DFP_abs(a::DFP{N}) where {N}
        # Check if a is an Error
        if DFP_isError(a)
            return a
        else
            return DFP{N}(0,a.expo,a.m)
        end
    end

    @inline function DFP_doublePrecision(a::DFP)
        prec = DFP_lengthMantissa(a)
        return DFP{2*prec}(a.s,a.expo,vcat(a.m,DFP_listbuilder(prec)))
    end

    function DFP_add(a::DFP,b::DFP)
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 4
            return a
        end
        if sa == 4
            return b
        end
        if fa == 3
            return a
        end
        if sa == 3
            return b
        end
        # If first arg is InF and sa == "0 or 1"then return InF
        if fa == 1 && (0 <= sa <= 1)
            return a
        end
        # If first arg is InF and sa is -InF
        if fa == 1 && sa == 2
            return DFP_createNaNError(precA)
        end
        # If first arg is -InF and sa is Number or -InF
        if fa == 2 && (sa == 0 || sa == 2)
            return a
        end
        # If first arg is -InF and sa is InF
        if fa == 2 && sa ==1
            return DFP_createNaNError(precA)
        end
        # if first arg is Number and sa is "InF or -InF"
        if fa == 0 && (1 <= sa <= 2)
            return b
        end
        DFP_setPrecision(DFP_genericAdd(DFP_doublePrecision(a),DFP_doublePrecision(b)),DFP_lengthMantissa(a))
    end

    function DFP_genericAdd(a::DFP{N},b::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        # Now we are sure that both sides have the same precision
        if DFP_isZero(a)
            return b
        end
        if DFP_isZero(b)
            return a
        end
        if a.s == 0 && b.s == 0 && a.expo == b.expo
            (r,c) = mda(a.m,b.m)
            if c == 0
                # no carry
                result = DFP{N}( 0, a.expo, r )
            else
                # with carry
                mantissa = r[1:end-1] # drop the last digit
                mantissa = vcat([c],mantissa) # join the carry digit
                result = DFP{N}( 0, a.expo + 1, mantissa) # increase expo by one
            end
            return result
        end # if a.s == 0 && b.s == 0 && a.expo == b.expo
        if a.s == 0 && b.s == 0 && a.expo > b.expo && a.expo - b.expo < precA
            expo_delta = a.expo - b.expo
            bdigits = vcat(DFP_listbuilder(expo_delta), b.m)
            bdigits = bdigits[1:end-expo_delta]
            (r,c) = mda(a.m,bdigits)
            if c == 0
                # no carry
                result = DFP{N}( 0, a.expo, r )
            else
                # with carry
                mantissa = r[1:end-1] # drop the last digit
                mantissa = vcat([c],mantissa) # join the carry digit
                result = DFP{N}( 0, a.expo + 1, mantissa) # increase expo by one
            end
            return result
        end # if a.s == 0 && b.s == 0 && a.expo > b.expo && a.expo - b.expo < precA
        if a.s == 0 && b.s == 0 && a.expo > b.expo && a.expo - b.expo >= precA
            return a
        end
        if a.s == 0 && b.s == 0 && a.expo < b.expo
            return DFP_genericAdd(b,a)
        end
        if a.s == 0 && b.s == 1 && a.expo == b.expo
            bdigits = ninecomplement(b.m)
            (r,c) = mda(a.m,bdigits)
            if c == 0
                # no carry
                mantissa = ninecomplement(r)
                if all( map(x->x==0,mantissa) )
                    # result of subtraction is zero
                    result = DFP{N}(0, 0, mantissa)
                else
                    # the result of subtraction is NEGATIVE
                    result = DFP_normalise( DFP{N}(1,a.expo,mantissa) )
                end # if sum(mantissa) == 0
            else
                # with carry
                one = DFP_listbuilder(precA)
                one[end] = 1
                (mantissa,c) = mda(r,one)
                # the result of subtraction is POSITIVE
                result = DFP_normalise( DFP{N}(0, a.expo, mantissa) )
            end
            return result
        end # if a.s == 0 && b.s == 1 && a.expo == b.expo
        if a.s == 0 && b.s == 1 && a.expo > b.expo && a.expo - b.expo < precA
            expo_delta = a.expo - b.expo
            bdigits = vcat(DFP_listbuilder(expo_delta),b.m)
            bdigits = bdigits[1:end-expo_delta]
            bdigits = ninecomplement(bdigits)
            (r,c) = mda(a.m,bdigits)
            if c == 0
                # no carry
                result = DFP_normalise(  DFP{N}( 0, a.expo, ninecomplement(r) )  )
            else
                # with carry
                one = DFP_listbuilder(precA)
                one[end] = 1
                (mantissa,c) = mda(r,one)
                result = DFP_normalise( DFP{N}(0, a.expo, mantissa) )
            end
            return result
        end # if a.s == 0 && b.s == 1 && a.expo > b.expo && a.expo - b.expo < precA
        if a.s == 0 && b.s == 1 && a.expo > b.expo && a.expo - b.expo >= precA
            return a
        end
        if a.s == 0 && b.s == 1 && a.expo < b.expo && b.expo - a.expo < precA
            expo_delta = b.expo - a.expo
            adigits = vcat(DFP_listbuilder(expo_delta),a.m)
            adigits = adigits[1:end-expo_delta]
            bdigits = ninecomplement(b.m)
            (r,c) = mda(adigits,bdigits)
            if c == 0
                # no carry
                result = DFP_normalise(  DFP{N}(1, b.expo, ninecomplement(r))  )
            else
                # with carry
                one = DFP_listbuilder(precA)
                one[end] = 1
                (mantissa,c) = mda(r, one)
                result = DFP_normalise(  DFP{N}(1, b.expo, mantissa)  )
            end
            return result
        end # if a.s == 0 && b.s == 1 && a.expo < b.expo && b.expo - a.expo < precA
        if a.s == 0 && b.s == 1 && a.expo < b.expo && b.expo - a.expo >= precA
            return b
        end
        if a.s == 1 && b.s == 0
            return DFP_genericAdd(b,a)
        end
        if a.s == 1 && b.s == 1
            r = DFP_genericAdd( DFP{N}(0,a.expo,a.m) , DFP{N}(0,b.expo,b.m) )
            return DFP{N}(1,r.expo,r.m)
        end
        println("We should never reached here")
        println("a is ",a)
        println("b is ",b)
        throw(ErrorException("Execution reach a part of the code that it should never reached in DFP_genericAdd"))
    end # function DFP_genericAdd

    @inline function DFP_sub(a::DFP{N},b::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 4
            return a
        end
        if sa == 4
            return b
        end
        if fa == 3
            return a
        end
        if sa == 3
            return b
        end
        # If first arg is InF and sa == "0 or 2"then return InF
        if fa == 1 && (sa == 0 || sa == 2)
            return a
        end
        # If first arg is InF and sa is InF
        if fa == 1 && sa == 1
            return DFP_createNaNError(precA)
        end
        # If first arg is -InF and sa is Number or InF
        if fa == 2 && (0 <= sa <= 1)
            return a
        end
        # If first arg is -InF and sa is -InF
        if fa == 2 && sa == 2
            return DFP_createNaNError(precA)
        end
        # if first arg is Number and sa is InF
        if fa == 0 && sa == 1
            return DFP_createNegInfError(precA)
        end
        # if first arg is Number and sa is -InF
        if fa == 0 && sa == 2
            return DFP_createInfError(precA)
        end
        if DFP_isZero(b)
            return a
        end
        DFP_add(a,DFP{N}((b.s+1)%2,b.expo,b.m))
    end

    @inline function DFP_genericSub(a::DFP{N},b::DFP{N}) where {N}
        if DFP_isZero(b)
            return a
        end
        DFP_genericAdd(a,DFP{N}((b.s+1)%2,b.expo,b.m))
    end

    function DFP_mul(a::DFP{N},b::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 4
            return a
        end
        if sa == 4
            return b
        end
        if fa == 3
            return a
        end
        if sa == 3
            return b
        end
        # If first arg is InF and sa == Num (0)
        if fa == 1 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createNegInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createInfError(precA)
            end
        end
        # If first arg is InF and sa == Inf (1)
        if fa == 1 && sa == 1
            return a
        end
        # If first arg is InF and sa == -Inf (2)
        if fa == 1 && sa == 2
            return b
        end
        # If first arg is -InF and sa == Num (0)
        if fa == 2 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createNegInfError(precA)
            end
        end
        # If first arg is -InF and sa == Inf (1)
        if fa == 2 && sa == 1
            return a
        end
        # If first arg is -InF and sa == -Inf (2)
        if fa == 2 && sa == 2
            return DFP_createInfError(precA)
        end
        # If first arg is Num and sa == Inf (1)
        if fa == 0 && sa == 1
            r = DFP_compare(a,DFP_createZero(precA))
            if r == -1
                return DFP_createNegInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createInfError(precA)
            end
        end
        # If first arg is Num and sa == -Inf (2)
        if fa == 0 && sa == 2
            r = DFP_compare(a,DFP_createZero(precA))
            if r == -1
                return DFP_createInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createNegInfError(precA)
            end
        end
        if DFP_isZero(a)
            return a
        end
        if DFP_isZero(b)
            return b
        end
        resultarray = IntegerDigits( FromDigits(a.m) * FromDigits(b.m) )
        len = length(resultarray)
        return DFP_setPrecision( DFP{len}( (a.s + b.s) % 2, a.expo - precA + b.expo - precB + length(resultarray) + 1, resultarray ), precA)
    end

    function DFP_genericMul(a::DFP,b::DFP)
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        if DFP_isZero(a)
            return a
        end
        if DFP_isZero(b)
            return b
        end
        resultarray = IntegerDigits( FromDigits(a.m) * FromDigits(b.m) )
        return DFP{precA}( (a.s + b.s) % 2, a.expo - precA + b.expo - precB + length(resultarray) + 1, resultarray[1:precA] )
    end

    # Division is tricky
    #
    # A ÷ B     we call A the dividend and we call B the divisor
    #
    # Given two integers a and b, with b ≠ 0, there exist unique
    # integers q and r such that
    #
    # A == B × q + r      where  0 <= r < abs(B)
    #
    # Notice that q can be negative but r is ALWAYS POSITIVE
    #
    # This is the definition of Euclidean Division
    #
    #  7 euclid_div  3 ==  2  with remainder  1
    #  7 euclid_div -3 == -2  with remainder  1
    # -7 euclid_div  3 == -3  with remainder  2
    # -7 euclid_div -3 ==  3  with remainder  2
    #
    # Julia division
    #
    #  7 div  3 ==  2  with remainder  1
    #  7 div -3 == -2  with remainder  1
    # -7 div  3 == -2  with remainder -1
    # -7 div -3 ==  2  with remainder -1
    #
    # fld division
    #
    #  7 fld  3 ==  2  with remainder  1
    #  7 fld -3 == -3  with remainder -2
    # -7 fld  3 == -3  with remainder  2
    # -7 fld -3 ==  2  with remainder -1
    #
    # cld division
    #
    #  7 cld  3 ==  3  with remainder -2
    #  7 cld -3 == -2  with remainder  1
    # -7 cld  3 == -2  with remainder -1
    # -7 cld -3 ==  3  with remainder  2

    function DFP_div(a::DFP{N},b::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 4
            return a
        end
        if sa == 4
            return b
        end
        if fa == 3
            return a
        end
        if sa == 3
            return b
        end
        # If first arg is ±Inf and sa == ±Inf
        if (fa == 1 || fa == 2) && (sa == 1 || sa == 2)
            return DFP_createNaNError(precA)
        end
        # If first arg is Num and sa == ±Inf
        if fa == 0  && (sa == 1 || sa == 2)
            return DFP_createZero(precA)
        end
        # If first arg is InF and sa == Num
        if fa == 1 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createNegInfError(precA)
            elseif r == 0
                return DFP_createInfError(precA)
            elseif r == 1
                return DFP_createInfError(precA)
            end
        end
        # If first arg is -InF and sa == Num
        if fa == 2 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createInfError(precA)
            elseif r == 0
                return DFP_createNegInfError(precA)
            elseif r == 1
                return DFP_createNegInfError(precA)
            end
        end
        # if b is zero then return DivByZeroError
        if DFP_isZero(b)
            if DFP_isZero(a)
                return DFP_createNaNError(precA)
            end
            if a.s == 0
                return DFP_createInfError(precA)
            else
                return DFP_createNegInfError(precA)
            end
        end
        # if a is zero then return zero
        if DFP_isZero(a)
            return a
        end
        newprec = precA + GUARD_DIGITS
        return DFP_setPrecision(
            DFP_genericDiv(
                DFP_setPrecision(a,newprec),
                DFP_setPrecision(b,newprec)
            )
        , precA)
    end

    function DFP_genericDiv(a::DFP{N},b::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        # Precision of LHS must be the same as RHS
        if precA != precB
            return DFP_createError(precA,2)
        end
        # if b is zero then return DivByZeroError
        if DFP_isZero(b)
            if DFP_isZero(a)
                return DFP_createNaNError(precA)
            end
            if a.s == 0
                return DFP_createInfError(precA)
            else
                return DFP_createNegInfError(precA)
            end
        end
        # if a is zero then return zero
        if DFP_isZero(a)
            return a
        end
        numerator = FromDigits(a.m)
        denominator = FromDigits(b.m)
        counter = precA + 1
        # if b is like one then adjust the exponent
        if b.m[1] == 1 && all( map(x->x==0,b.m[2:end]) )
            return DFP{N}( (a.s+b.s)%2, a.expo - b.expo, a.m)
        end
        # Next we obtain counter number of digits
        digitlist = Int8[]
        while counter > 0
            if numerator < denominator
                # numerator < denominator
                push!(digitlist, 0)
            else
                # numerator >= denominator
                newdigit = div(numerator, denominator)
                push!(digitlist, newdigit)
                numerator -= (newdigit * denominator)
            end
            numerator *= 10
            counter -= 1
        end # while counter > 0
        if digitlist[1] == 0
            newexponent = a.expo - b.expo - 1
            popfirst!(digitlist)
        else
            newexponent = a.expo - b.expo
            pop!(digitlist)
        end
        return DFP{N}((a.s + b.s) % 2, newexponent, digitlist)
    end

    # Function DFP_compare : Compare a with b
    # return Int -1 if a < b
    # return Int  0  if a == b
    # return Int  1  if a > b
    @inline function DFP_compare(a::DFP{N},b::DFP{N})::Int64 where {N}
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 3 || fa == 4 || sa == 3 || sa == 4
            return typemin(Int64)
        end
        if fa == 0
            if sa == 1
                return -1
            elseif sa == 2
                return 1
            end
        elseif fa == 1
            if sa == 0
                return 1
            end
            if sa == 1
                return 0
            end
            if sa == 2
                return -1
            end
        elseif fa == 2
            if sa == 0
                return -1
            end
            if sa == 1
                return -1
            end
            if sa == 2
                return 0
            end
        end
        result = DFP_genericSub(a,b)
        if DFP_isZero(result)
            return 0
        end
        if result.s == 1
            return -1
        end
        return 1
    end

    function DFP_fma(a::DFP{N},b::DFP{N},c::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        precC = DFP_lengthMantissa(c)
        # Precision of LHS must be the same as RHS
        if precA != precB || precA != precC
            return DFP_createError(precA,2)
        end
        (fa,sa) = DFP_lhsrhs_kind(a,b)
        # Check if a or b is an Ordinary Error or NaN
        if fa == 4
            return a
        end
        if sa == 4
            return b
        end
        if fa == 3
            return a
        end
        if sa == 3
            return b
        end
        # If first arg is InF and sa == Num (0)
        if fa == 1 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createNegInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createInfError(precA)
            end
        end
        # If first arg is InF and sa == Inf (1)
        if fa == 1 && sa == 1
            return a
        end
        # If first arg is InF and sa == -Inf (2)
        if fa == 1 && sa == 2
            return b
        end
        # If first arg is -InF and sa == Num (0)
        if fa == 2 && sa == 0
            r = DFP_compare(b,DFP_createZero(precA))
            if r == -1
                return DFP_createInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createNegInfError(precA)
            end
        end
        # If first arg is -InF and sa == Inf (1)
        if fa == 2 && sa == 1
            return a
        end
        # If first arg is -InF and sa == -Inf (2)
        if fa == 2 && sa == 2
            return DFP_createInfError(precA)
        end
        # If first arg is Num and sa == Inf (1)
        if fa == 0 && sa == 1
            r = DFP_compare(a,DFP_createZero(precA))
            if r == -1
                return DFP_createNegInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createInfError(precA)
            end
        end
        # If first arg is Num and sa == -Inf (2)
        if fa == 0 && sa == 2
            r = DFP_compare(a,DFP_createZero(precA))
            if r == -1
                return DFP_createInfError(precA)
            elseif r == 0
                return DFP_createNaNError(precA)
            elseif r == 1
                return DFP_createNegInfError(precA)
            end
        end
        if DFP_isZero(a) || DFP_isZero(b)
            return c
        end
        resultarray = IntegerDigits( FromDigits(a.m) * FromDigits(b.m) )
        highprec = length(resultarray)
        temp = DFP{highprec}( (a.s + b.s) % 2, a.expo - precA + b.expo - precB + length(resultarray) + 1, resultarray )
        newc = DFP_setPrecision(c,highprec)
        result = DFP_setPrecision(DFP_genericAdd(temp,newc),precA)
        return result
    end

    """
        DFP_genericFma(a::DFP,b::DFP,c::DFP)

    Return the fused multipl add function. This gives the
    best most accurate result of a * b + c
    """
    function DFP_genericFma(a::DFP{N},b::DFP{N},c::DFP{N}) where {N}
        precA = DFP_lengthMantissa(a)
        precB = DFP_lengthMantissa(b)
        precC = DFP_lengthMantissa(c)
        # Precision of LHS must be the same as RHS
        if precA != precB || precA != precC
            return DFP_createError(precA,2)
        end
        if DFP_isZero(a) || DFP_isZero(b)
            return c
        end
        resultarray = IntegerDigits( FromDigits(a.m) * FromDigits(b.m) )
        highprec = length(resultarray)
        temp = DFP{highprec}( (a.s + b.s) % 2, a.expo - precA + b.expo - precB + length(resultarray) + 1, resultarray )
        newc = DFP_setPrecision(c,highprec)
        result = DFP_setPrecision(DFP_genericAdd(temp,newc),precA)
        return result
    end

    @inline function DFP_integer(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if a.expo < 0
            return DFP_createZero(prec)
        end
        if a.expo  > prec - 2
            return a
        end
        # Otherwise 0 <= a.expo < prec - 1
        padlen = prec - a.expo - 1
        newmantissa = vcat(a.m[1:a.expo+1],DFP_listbuilder(padlen))
        return DFP{prec}(a.s, a.expo, newmantissa)
    end

    @inline function DFP_fraction(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if a.expo < 0
            return a
        end
        if a.expo  > prec - 2
            return DFP_createZero(prec)
        end
        # Otherwise 0 <= a.expo < prec - 1
        padlen = a.expo + 1
        newmantissa = vcat(a.m[a.expo+2:end],DFP_listbuilder(padlen))
        return DFP_normalise( DFP{prec}(a.s, a.expo - padlen, newmantissa) )
    end

    @inline function DFP_leftShift(a::DFP,n::Integer)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        if n == 0
            return a
        end
        prec = DFP_lengthMantissa(a)
        if n >= prec
            return DFP_createZero(prec)
        end
        newexpo = a.expo - n
        newmantissa = vcat(a.m[n+1:end],zeros(Int8,n))
        if all(  map(x->x==0,newmantissa)  )
            return DFP_createZero(prec)
        end
        return DFP_normalise(  DFP{prec}(a.s,newexpo,newmantissa)  )
    end

    @inline function DFP_rightShift(a::DFP,n::Integer)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        if n == 0
            return a
        end
        prec = DFP_lengthMantissa(a)
        if n >= prec
            return DFP_createZero(prec)
        end
        newmantissa = vcat(a.m[1:end-n],zeros(Int8,n))
        if all(  map(x->x==0,newmantissa)  )
            return DFP_createZero(prec)
        end
        return DFP_normalise(  DFP{prec}(a.s,a.expo,newmantissa)  )
    end

    @inline function DFP_leftCircularShift(a::DFP,n::Integer)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        if n == 0
            return a
        end
        prec = DFP_lengthMantissa(a)
        return DFP_normalise( DFP{prec}(a.s,a.expo,circshift(a.m,-n)) )
    end

    @inline function DFP_rightCircularShift(a::DFP,n::Integer)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        if n == 0
            return a
        end
        prec = DFP_lengthMantissa(a)
        return DFP_normalise( DFP{prec}(a.s,a.expo,circshift(a.m,n)) )
    end


    function DFP_euclidInteger(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if a.s == 0
            # The number is positive
            if a.expo < 0
                return DFP_createZero(prec)
            end
            if a.expo  > prec - 2
                return a
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = prec - a.expo - 1
            newmantissa = vcat(a.m[1:a.expo+1],DFP_listbuilder(padlen))
            return DFP{prec}(a.s, a.expo, newmantissa)
        else
            # The number is negative
            if a.expo < 0
                return DFP_createMinusOne(prec)
            end
            if a.expo  > prec - 2
                return a
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = prec - a.expo - 1
            if padlen > 0
                fractionalmantissa = a.m[a.expo+2:end]
            else
                fractionalmantissa = Int8[]
            end
            newmantissa = vcat(a.m[1:a.expo+1],DFP_listbuilder(padlen))
            if any(  map(x->x>0,fractionalmantissa)  )
                return DFP{prec}(a.s, a.expo, newmantissa) + DFP_createMinusOne(prec)
            else
                return DFP{prec}(a.s, a.expo, newmantissa)
            end
        end
    end

    function DFP_euclidFraction(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if a.s == 0
            # The number is positive
            if a.expo < 0
                return a
            end
            if a.expo  > prec - 2
                return DFP_createZero(prec)
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = a.expo + 1
            newmantissa = vcat(a.m[a.expo+2:end],DFP_listbuilder(padlen))
            return DFP{prec}(0, a.expo - padlen, newmantissa)
        else
            # The number is negative
            if a.expo < 0
                return DFP_createOne(prec) + a
            end
            if a.expo  > prec - 2
                return DFP_createZero(prec)
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = a.expo + 1
            newmantissa = vcat(a.m[a.expo+2:end],DFP_listbuilder(padlen))
            return DFP_createOne(prec) + DFP{prec}(a.s, a.expo - padlen, newmantissa)
        end
    end

    function DFP_ceiling(a::DFP)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        prec = DFP_lengthMantissa(a)
        if a.s == 0
            # The number is positive
            if a.expo < 0
                return DFP_createOne(prec)
            end
            if a.expo  > prec - 2
                return a
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = prec - a.expo - 1
            if padlen > 0
                fractionalmantissa = a.m[a.expo+2:end]
            else
                fractionalmantissa = Int8[]
            end
            newmantissa = vcat(a.m[1:a.expo+1],DFP_listbuilder(padlen))
            if any(  map(x->x>0,fractionalmantissa)  )
                return DFP{prec}(a.s, a.expo, newmantissa) + DFP_createOne(prec)
            else
                return DFP{prec}(a.s, a.expo, newmantissa)
            end
        else
            # The number is negative
            if a.expo < 0
                return DFP_createZero(prec)
            end
            if a.expo  > prec - 2
                return a
            end
            # Otherwise 0 <= a.expo < prec - 1
            padlen = prec - a.expo - 1
            newmantissa = vcat(a.m[1:a.expo+1],DFP_listbuilder(padlen))
            return DFP{prec}(a.s, a.expo, newmantissa)
        end
    end

    function DFP_Pow2(a::DFP{N}) where {N}
        dpA = DFP_doublePrecision(a)
        DFP_setPrecision( DFP_genericMul(dpA,dpA) , N )
    end

    function DFP_Pow4(a::DFP{N}) where {N}
        dpA = DFP_doublePrecision(a)
        temp = DFP_genericMul(dpA,dpA)
        temp = DFP_genericMul(temp,temp)
        DFP_setPrecision( temp , N )
    end

    function DFP_Pow8(a::DFP{N}) where {N}
        dpA = DFP_doublePrecision(a)
        temp = DFP_genericMul(dpA,dpA)
        temp = DFP_genericMul(temp,temp)
        temp = DFP_genericMul(temp,temp)
        DFP_setPrecision( temp , N )
    end

    function DFP_Pow10(a::DFP{N}) where {N}
        dpA = DFP_doublePrecision(a)
        two = DFP_genericMul(dpA,dpA)
        four = DFP_genericMul(two,two)
        eight = DFP_genericMul(four,four)
        ten = DFP_genericMul(two,eight)
        DFP_setPrecision( ten , N )
    end

    @inline function DFP_getUnitDigit(a::DFP)::Union{Int64,Nothing}
        # Check if a is an Error
        if DFP_isError(a)
            return nothing
        end
        prec = DFP_lengthMantissa(a)
        pos = a.expo + 1
        if pos < 1 || pos > prec
            return nothing
        else
            return a.m[pos]
        end
    end

    function DFP_altSqrt(a::DFP)
        prec   = DFP_lengthMantissa(a)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        # square root of a negative number is an error
        if a.s == 1
            return DFP_createError(prec,3)
        end
        # square root of zero is zero
        if DFP_isZero(a)
            return a
        end
        # square root of one is one
        if DFP_isOne(a)
            return a
        end
        DFP_setPrecision(
            DFP_altSqrtBackend(
                DFP_setPrecision(a,prec+2)
            ),
            prec
        )
    end

    function DFP_altSqrtBackend(a::DFP)
        prec = DFP_lengthMantissa(a)
        if isodd(a.expo)
            mantissa = vcat(  a.m, DFP_listbuilder(  prec + 2  )  )
            outputexp = (a.expo - 1) / 2
        else
            mantissa = vcat( Int8[0], a.m, DFP_listbuilder(  prec + 2  )  )
            outputexp = a.expo / 2
        end
        resultlist = Int8[]
        root = BigInt(0)
        remainder = BigInt(0)
        counter = BigInt(0)
        while counter < prec + 1
            remainder = remainder * 100 + mantissa[1]*10 + mantissa[2]
            mantissa = mantissa[3:end]
            if 5 * (20 * root + 5 ) > remainder
                guess = 4
                while guess * (20 * root + guess) > remainder
                    guess = guess - 1
                end
            else
                guess = 9
                while guess * (20 * root + guess) > remainder
                    guess = guess - 1
                end
            end
            push!(resultlist, guess)
            remainder = remainder - guess * (20 * root + guess)
            root = root * 10 + guess
            counter = counter + 1
        end
        M = length(resultlist)
        DFP_setPrecision(DFP{M}(0, outputexp, resultlist), prec  )
    end

    function DFP_altLog10(a::DFP)
        prec = DFP_lengthMantissa(a)
        # Check if a is an Error
        if DFP_isError(a)
            return a
        end
        # Logarithmn of a negative number is an error
        if a.s == 1
            return DFP_createError(prec,3)
        end
        # logarithmn of one is zero
        if DFP_isOne(a)
            return DFP_createZero(prec)
        end
        DFP_setPrecision(
            DFP_altLog10Backend(
                DFP_setPrecision(a,prec+2)
            ),
            prec
        )
    end

    function DFP_altLog10Backend(a::DFP)
        prec = DFP_lengthMantissa(a)
        N = prec
        data = Int8[]
        # Next turn exponent into DFP
        digits = IntegerDigits( a.expo )
        if a.expo == 0
            intfpo = DFP_createZero(prec)
        elseif a.expo < 0
            intfpo = DFP_setPrecision(DFP(1, length(digits) - 1, digits), prec)
        else
            intfpo = DFP_setPrecision(DFP(0, length(digits) - 1, digits), prec)
        end
        # Next change the exponent to zero
        num = DFP{prec}(a.s,0,a.m)
        # Create Ten
        ten = DFP_createTen(prec)
        # num = a^10
        num = DFP_Pow10(num)
        # To be filled in later
        while prec > 0
            prec -= 1
            digit = 0
            while DFP_compare(num,ten) >= 0
                digit += 1
                num = DFP{N}(num.s,num.expo - 1,num.m)
            end
            push!(data,digit)
            num = DFP_Pow10(num)
        end
        intfpo + DFP{N}(0,-1,data)
    end

    function DFP_lambertW(a::DFP)
        loopflag = true
        counter = 0
        prec = DFP_lengthMantissa(a)
        newprec = max(10,prec + 2)
        xvalue = DFP_setPrecision(a,newprec)
        one = DFP_createOne(newprec)
        result = deepcopy(one)
        two = DFP_createTwo(newprec)
        v = DFP_createZero(newprec)
        olddiff = one
        newdiff = v
        while loopflag && counter < 256
            counter += 1
            v = result
            evalue = exp(result)
            f = (result * evalue) - xvalue
            temp = f / ( ( result * two ) + two )
            temp = temp * ( result + two )
            den = ( evalue * (result + one) ) - temp
            result = result - (f / den)
            if result == v || olddiff == newdiff
                loopflag = false
            else
                olddiff = newdiff
                newdiff = abs(result - v)
            end
        end
        DFP_setPrecision(result,prec)
    end

    # Demo/Debug version of lambertW function
    function DFP_lambertW_demo(a::DFP)
        loopflag = true
        counter = 0
        prec = DFP_lengthMantissa(a)
        newprec = max(10,prec + 2)
        xvalue = DFP_setPrecision(a,newprec)
        one = DFP_createOne(newprec)
        result = deepcopy(one)
        two = DFP_createTwo(newprec)
        v = DFP_createZero(newprec)
        olddiff = one
        newdiff = v
        while loopflag && counter < 256
            counter += 1
            v = result
            evalue = exp(result)
            f = (result * evalue) - xvalue
            temp = f / ( ( result * two ) + two )
            temp = temp * ( result + two )
            den = ( evalue * (result + one) ) - temp
            result = result - (f / den)
            if result == v || olddiff == newdiff
                loopflag = false
            else
                olddiff = newdiff
                newdiff = abs(result - v)
                println("The difference is ",DFP_toShortCommonString( newdiff ) )
            end
        end
        println("The value of counter is ",counter)
        DFP_setPrecision(result,prec)
    end

    # There are two branches for the Lambert W function
    # this is the value of the second branch
    function DFP_lambertWsb(a::DFP)
        loopflag = true
        counter = 0
        prec = DFP_lengthMantissa(a)
        newprec = max(10,prec + 2)
        xvalue = DFP_setPrecision(a,newprec)
        one = DFP_createOne(newprec)
        two = DFP_createTwo(newprec)
        minusone = DFP_createMinusOne(newprec)
        result = two * minusone
        v = DFP_createZero(newprec)
        olddiff = one
        newdiff = v
        while loopflag && counter < 256
            counter += 1
            v = result
            evalue = exp(result)
            f = (result * evalue) - xvalue
            temp = f / ( ( result * two ) + two )
            temp = temp * ( result + two )
            den = ( evalue * (result + one) ) - temp
            result = result - (f / den)
            if result == v || olddiff == newdiff
                loopflag = false
            else
                olddiff = newdiff
                newdiff = abs(result - v)
            end
        end
        DFP_setPrecision(result,prec)
    end

    function DFP_log10Gamma(a::DFP)
        prec = DFP_lengthMantissa(a)
        newprec = prec + GUARD_DIGITS
        ten = DFP_createTen(newprec)
        newa = DFP_setPrecision(a,newprec)
        DFP_setPrecision(loggamma(newa)/log(ten),prec)
    end

    function DFP_log10Factorial(a::DFP)
        prec = DFP_lengthMantissa(a)
        one = DFP_createOne(prec)
        DFP_log10Gamma(a + one)
    end

    function DFP_Perm(n::DFP{N},r::DFP{N},expansion::Int64=20) where {N}
        newprec = expansion * N
        nminusr = n - r
        newn = DFP_setPrecision(n,newprec)
        newnminusr = DFP_setPrecision(nminusr,newprec)
        temp = DFP_log10Factorial(newn) - DFP_log10Factorial(newnminusr)
        DFP_setPrecision(exp10(temp),N)
    end

    function DFP_Comb(n::DFP{N},r::DFP{N},expansion::Int64=20) where {N}
        newprec = expansion * N
        nminusr = n - r
        newn = DFP_setPrecision(n,newprec)
        newnminusr = DFP_setPrecision(nminusr,newprec)
        newr = DFP_setPrecision(r,newprec)
        temp = DFP_log10Factorial(newn) - DFP_log10Factorial(newnminusr) - DFP_log10Factorial(newr)
        DFP_setPrecision(exp10(temp),N)
    end

    # return 0 if error
    # return 1 if number is zero or positive
    # otherwise return -1
    @inline function DFP_sideStep(num::DFP)
        # Check if a is an Error
        if DFP_isError(num)
            return 0
        end
        prec = DFP_getPrecision(num)
        zero = DFP_createZero(prec)
        if DFP_compare(num,zero) >= 0
            return 1
        else
            return -1
        end
    end

    # return 0 if number is zero
    # return 1 if number is positive
    # return -1 if number is DFP_isNegative
    @inline function DFP_sgn(num::DFP)
        # Check if a is an Error
        if DFP_isError(num)
            return num
        end
        prec = DFP_getPrecision(num)
        zero = DFP_createZero(prec)
        if DFP_compare(num,zero) == 0
            return 0
        elseif DFP_compare(num,zero) > 0
            return 1
        else
            return -1
        end
    end

    function DFP_sign(num::DFP)
        return DFP_sgn(num)
    end

    function DFP_forwardEpsilon(value::DFP)
        # Check if a is an Error
        if DFP_isError(value)
            return value
        end
        if DFP_isZero(value)
            prec = DFP_lengthMantissa(value)
            return DFP_forwardEpsilon(DFP_createOne(prec))
        end
        if value.s == 0
            nextpoint = DFP_roundup(value)
        else
            nextpoint = DFP_rounddown(value)
        end
        return nextpoint - value
    end

    function DFP_backwardEpsilon(value::DFP)
        # Check if a is an Error
        if DFP_isError(value)
            return value
        end
        if DFP_isZero(value)
            prec = DFP_lengthMantissa(value)
            return DFP_backwardEpsilon(DFP_createOne(prec))
        end
        if value.s == 0
            prevpoint = DFP_rounddown(value)
        else
            prevpoint = DFP_roundup(value)
        end
        return value - prevpoint
    end

    function DFP_bidirectionalEpsilon(value::DFP)
        # Check if a is an Error
        if DFP_isError(value)
            return value
        end
        if DFP_isZero(value)
            prec = DFP_lengthMantissa(value)
            return DFP_bidirectionalEpsilon(DFP_createOne(prec))
        end
        if value.s == 0
            nextpoint = DFP_roundup(value)
            prevpoint = DFP_rounddown(value)
        else
            nextpoint = DFP_rounddown(value)
            prevpoint = DFP_roundup(value)
        end
        forward_interval = nextpoint - value
        backward_interval = value - prevpoint
        return max(forward_interval,backward_interval)
    end

    # Generate random DFP with exporange eg -5:5
    #
    # Please note that this function returns a random
    # DFP number which is NOT UNIFORMLY random in the
    # real number domain because floating point number
    # is NOT evenly distributed in the real number domain
    function DFP_randomDFP(exporange,prec)
        prec > 0 ? newmantissa = [ rand(1:9) ] : newmantissa = []
        if prec > 1
            append!(newmantissa, rand(0:9,prec-1))
        end
        DFP{prec}(rand(0:1),BigInt(rand(exporange)),newmantissa)
    end

    function DFP_randomDFP(exporange)
        prec = DEFAULT_PRECISION[][Base.Threads.threadid()]
        prec > 0 ? newmantissa = [ rand(1:9) ] : newmantissa = []
        if prec > 1
            append!(newmantissa, rand(0:9,prec-1))
        end
        DFP{prec}(rand(0:1),BigInt(rand(exporange)),newmantissa)
    end

    # Return in DFP form the result of a n sided dice. ie 1-6 for n = 6
    function DFP_randomDFPinteger(n::Int64)
        if n == 0
            return DFP(1)
        else
            result = DFP_integer(DFP_randomDFPbetweenZeroandOneSafe() * DFP(abs(n))) + DFP(1)
            if n > 0
                return result
            else
                # Put the value into negative form
                prec = DEFAULT_PRECISION[][Base.Threads.threadid()]
                return DFP{prec}(1,result.expo,result.m)
            end
        end
    end

    # Return in DFP form the result of a n sided dice. ie 1-6 for n = 6
    function DFP_randomDFPinteger(n::Int64,prec::Int64)
        safeprec = prec < 0 ? 0 : prec
        if n == 0
            return DFP(1,safeprec)
        else
            result = DFP_integer(DFP_randomDFPbetweenZeroandOneSafe(safeprec) * DFP(abs(n),safeprec)) + DFP(1,safeprec)
            if n > 0
                return result
            else
                # Put the value into negative form
                return DFP{safeprec}(1,result.expo,result.m)
            end
        end
    end

    # Return in DFP form the range
    function DFP_randomDFPinteger(range::UnitRange{Int64})
        lo = first(range)
        hi = last(range)
        n = hi - lo
        if n == 0
            return DFP(lo)
        else
            result = DFP_integer(DFP_randomDFPbetweenZeroandOneSafe() * DFP(abs(n)+1)) + DFP(lo)
            if n > 0
                return result
            else
                # Put the value into negative form
                prec = DFP_lengthMantissa(result)
                return DFP{prec}(1,result.expo,result.m)
            end
        end
    end

    # Return in DFP form the range
    function DFP_randomDFPinteger(range::UnitRange{Int64},prec::Int64)
        safeprec = prec < 0 ? 0 : prec
        lo = first(range)
        hi = last(range)
        n = hi - lo
        if n == 0
            return DFP(lo,safeprec)
        else
            result = DFP_integer(DFP_randomDFPbetweenZeroandOneSafe(safeprec) * DFP(abs(n)+1,safeprec)) + DFP(lo,safeprec)
            if n > 0
                return result
            else
                # Put the value into negative form
                return DFP{safeprec}(1,result.expo,result.m)
            end
        end
    end

    function DFP_randomDFPbetweenZeroandOne(prec)
        if prec > 0
            newmantissa = rand(0:9,prec)
            if newmantissa[1]==0
                # if the first digit is zero
                if all(  map(x->x==0,newmantissa)  )
                    # if all the digits are zero then
                    # return zero
                    return DFP{prec}(0,0,newmantissa)
                else
                    # Otherwise normalise the number
                    return DFP_normalise( DFP{prec}(0,-1,newmantissa) )
                end
            end
            # Othersie return the number with the exponent set to -1
            return DFP{prec}(0,-1,newmantissa)
        else
            # Othersie return the number with the exponent set to 0
            return DFP{0}(0,0,[])
        end
    end

    function DFP_randomDFPbetweenZeroandOne()
        prec = DEFAULT_PRECISION[][Base.Threads.threadid()]
        return DFP_randomDFPbetweenZeroandOne(prec)
    end

    # Same as before but using a model DFP for prec
    function DFP_randomDFPbetweenZeroandOne_likeModel(model::DFP)
        DFP_randomDFPbetweenZeroandOne(DFP_lengthMantissa(model))
    end

    function DFP_randomDFPbetweenZeroandOneSafe(prec)
        if prec < 1
            return DFP{0}(0,0,[])
        end
        result = DFP_randomDFPbetweenZeroandOne(prec)
        while DFP_isZero(result)
            result = DFP_randomDFPbetweenZeroandOne(prec)
        end
        return result
    end

    function DFP_randomDFPbetweenZeroandOneSafe()
        prec = DEFAULT_PRECISION[][Base.Threads.threadid()]
        return DFP_randomDFPbetweenZeroandOneSafe(prec)
    end

    # Same as before but using a model DFP for prec
    function DFP_randomDFPbetweenZeroandOneSafe_likeModel(model::DFP)
        DFP_randomDFPbetweenZeroandOneSafe(DFP_lengthMantissa(model))
    end

    # Obtain the derivative using Numerical Differentiation
    function DFP_derivative(func::Function,v::DFP)
        oldprec = DFP_getPrecision(v)
        newprec = oldprec * 4
        value = DFP_setPrecision(v,newprec)
        if value.s == 0
            nextpoint = DFP_roundup(value)
            prevpoint = DFP_rounddown(value)
        else
            nextpoint = DFP_rounddown(value)
            prevpoint = DFP_roundup(value)
        end
        forward_interval = nextpoint - value
        backward_interval = value - prevpoint
        # Set the interval size to be (1 + (vprec ÷ 3)) magnitude below the value
        interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 3)) ) * max(forward_interval,backward_interval)
        # println("Interval is ",interval)
        # first derivative is (1/(2*d))*(f(x+d) - f(x-d))
        result = (func(value + interval) - func(value - interval))/(interval + interval)
        # println("Result is ",result)
        return DFP_setPrecision(result,oldprec)
    end

    function DFP_firstsecondDerivative(func::Function,v::DFP)
        oldprec = DFP_getPrecision(v)
        newprec = oldprec * 4
        value = DFP_setPrecision(v,newprec)
        if value.s == 0
            nextpoint = DFP_roundup(value)
            prevpoint = DFP_rounddown(value)
        else
            nextpoint = DFP_rounddown(value)
            prevpoint = DFP_roundup(value)
        end
        forward_interval = nextpoint - value
        backward_interval = value - prevpoint
        newprec = DFP_getPrecision(value)
        # Set the interval size to be (1 + (newprec ÷ 3)) magnitude below the value
        interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 3)) ) * max(forward_interval,backward_interval)
        twointerval = interval + interval
        p0 = func(value)
        np1 = func(value + interval)
        np2 = func(value + twointerval)
        pp1 = func(value - interval)
        pp2 = func(value - twointerval)
        first  = (np1 - pp1)/twointerval
        num = np2 - 2 * p0 + pp2
        den = 4 * interval * interval
        second = num / den
        first = DFP_setPrecision(first,oldprec)
        second = DFP_setPrecision(second,oldprec)
        return  (first,second)
    end

    function DFP_firstsecondthirdDerivative(func::Function,v::DFP)
        oldprec = DFP_getPrecision(v)
        newprec = oldprec * 4
        value = DFP_setPrecision(v,newprec)
        if value.s == 0
            nextpoint = DFP_roundup(value)
            prevpoint = DFP_rounddown(value)
        else
            nextpoint = DFP_rounddown(value)
            prevpoint = DFP_roundup(value)
        end
        forward_interval = nextpoint - value
        backward_interval = value - prevpoint
        newprec = DFP_getPrecision(value)
        # Set the interval size to be (1 + (newprec ÷ 3)) magnitude below the value
        interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 3)) ) * max(forward_interval,backward_interval)

        twointerval = interval + interval
        threeinterval = twointerval + interval
        p0 = func(value)
        np1 = func(value + interval)
        np2 = func(value + twointerval)
        np3 = func(value + threeinterval)
        pp1 = func(value - interval)
        pp2 = func(value - twointerval)
        pp3 = func(value - threeinterval)
        first  = (np1 - pp1)/twointerval
        num = np2 - 2 * p0 + pp2
        den = 4 * interval * interval
        second = num / den
        num = np3 - 3 * np1 + 3 * pp1 - pp3
        den = 8 * interval * interval * interval
        third = num / den
        first = DFP_setPrecision(first,oldprec)
        second = DFP_setPrecision(second,oldprec)
        third = DFP_setPrecision(third,oldprec)
        return  (first,second,third)
    end

    # FuncType = "MultiParameters" or "VectorParameters"
    function DFP_derivative_multivariate(func::Function,varnumvec::Vector{Int64},value::Vector{DFP{N}};FuncType::String="MultiParameters") where {N}
        local len = length(varnumvec)
        local derivative
        local nextpoint
        local prevpoint
        local forward_interval
        local backward_interval
        local interval
        MultiParam_flag = FuncType == "MultiParameters"
        # Special case, we reach the end of the derivative chain
        if len == 0
            return DFP{N}(0)
        end
        if len == 1
            local i = varnumvec[1]
            v = value[i]
            oldprec = DFP_getPrecision(v)
            newprec = oldprec * 2
            v2 = DFP_setPrecision(v,newprec)
            if v2.s == 0
                nextpoint = DFP_roundup(v2)
                prevpoint = DFP_rounddown(v2)
            else
                nextpoint = DFP_rounddown(v2)
                prevpoint = DFP_roundup(v2)
            end
            forward_interval = nextpoint - v2
            backward_interval = v2 - prevpoint
            # interval = DFP(10)^( DFP_getPrecision(v) ÷ 2 ) * max(forward_interval,backward_interval)
            interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 2)) ) * max(forward_interval,backward_interval)
            local args1
            local args2
            args1 = Array{DFP{newprec},1}()
            args2 = Array{DFP{newprec},1}()
            n = length(value)
            for k = 1:n
                push!(args1, DFP_setPrecision(value[k],newprec))
                push!(args2, DFP_setPrecision(value[k],newprec))
            end
            args1[i] = v2 + interval
            args2[i] = v2 - interval
            derivative = MultiParam_flag ? (func(args1...) - func(args2...))/(interval + interval) : (func(args1) - func(args2))/(interval + interval)
            return DFP_setPrecision(derivative,oldprec)
        end
        if len > 1
            newvarnumvec = deepcopy(varnumvec)
            i = pop!(newvarnumvec)
            v = value[i]
            oldprec = DFP_getPrecision(v)
            newprec = oldprec * 2
            v2 = DFP_setPrecision(v,newprec)
            if v2.s == 0
                nextpoint = DFP_roundup(v2)
                prevpoint = DFP_rounddown(v2)
            else
                nextpoint = DFP_rounddown(v2)
                prevpoint = DFP_roundup(v2)
            end
            forward_interval = nextpoint - v2
            backward_interval = v2 - prevpoint
            # interval = DFP(10)^( DFP_getPrecision(v) ÷ 2 ) * max(forward_interval,backward_interval)
            interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 2)) ) * max(forward_interval,backward_interval)
            local newvalue1
            local newvalue2
            newvalue1 = Array{DFP{newprec},1}()
            newvalue2 = Array{DFP{newprec},1}()
            n = length(value)
            for k = 1:n
                push!(newvalue1, DFP_setPrecision(value[k],newprec))
                push!(newvalue2, DFP_setPrecision(value[k],newprec))
            end
            newvalue1[i] = v2 + interval
            newvalue2[i] = v2 - interval
            derivative = (DFP_derivative_multivariate(func,newvarnumvec,newvalue1; :FuncType => FuncType) -  DFP_derivative_multivariate(func,newvarnumvec,newvalue2; :FuncType => FuncType))/(interval + interval)
            return DFP_setPrecision(derivative,oldprec)
        end
    end

    # FuncType = "MultiParameters" or "VectorParameters"
    function DFP_derivative_multifunc_multivariate(func::Function,funcnum::Int64,varnumvec::Vector{Int64},value::Vector{DFP{N}};FuncType::String="MultiParameters") where {N}
        local len = length(varnumvec)
        local derivative
        local nextpoint
        local prevpoint
        local forward_interval
        local backward_interval
        local interval
        MultiParam_flag = FuncType == "MultiParameters"
        # Special case, we reach the end of the derivative chain
        if len == 0
            return DFP(0)
        end
        if len == 1
            local i = varnumvec[1]
            v = value[i]
            oldprec = DFP_getPrecision(v)
            newprec = oldprec * 2
            v2 = DFP_setPrecision(v,newprec)
            if v2.s == 0
                nextpoint = DFP_roundup(v2)
                prevpoint = DFP_rounddown(v2)
            else
                nextpoint = DFP_rounddown(v2)
                prevpoint = DFP_roundup(v2)
            end
            forward_interval = nextpoint - v2
            backward_interval = v2 - prevpoint
            # interval = DFP(10)^( DFP_getPrecision(v) ÷ 2 ) * max(forward_interval,backward_interval)
            interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 2)) ) * max(forward_interval,backward_interval)
            local args1
            local args2
            args1 = Array{DFP{newprec},1}()
            args2 = Array{DFP{newprec},1}()
            n = length(value)
            for k = 1:n
                push!(args1, DFP_setPrecision(value[k],newprec))
                push!(args2, DFP_setPrecision(value[k],newprec))
            end
            args1[i] = v2 + interval
            args2[i] = v2 - interval
            derivative = MultiParam_flag ? (func(args1...)[funcnum] - func(args2...)[funcnum])/(interval + interval) : (func(args1)[funcnum] - func(args2)[funcnum])/(interval + interval)
            return DFP_setPrecision(derivative,oldprec)
        end
        if len > 1
            newvarnumvec = deepcopy(varnumvec)
            i = pop!(newvarnumvec)
            v = value[i]
            oldprec = DFP_getPrecision(v)
            newprec = oldprec * 2
            v2 = DFP_setPrecision(v,newprec)
            if v2.s == 0
                nextpoint = DFP_roundup(v2)
                prevpoint = DFP_rounddown(v2)
            else
                nextpoint = DFP_rounddown(v2)
                prevpoint = DFP_roundup(v2)
            end
            forward_interval = nextpoint - v2
            backward_interval = v2 - prevpoint
            # interval = DFP(10)^( DFP_getPrecision(v) ÷ 2 ) * max(forward_interval,backward_interval)
            interval = DFP(10,newprec)^( newprec - (1 + (newprec ÷ 2)) ) * max(forward_interval,backward_interval)
            local newvalue1
            local newvalue2
            newvalue1 = Array{DFP{newprec},1}()
            newvalue2 = Array{DFP{newprec},1}()
            n = length(value)
            for k = 1:n
                push!(newvalue1, DFP_setPrecision(value[k],newprec))
                push!(newvalue2, DFP_setPrecision(value[k],newprec))
            end
            newvalue1[i] = v2 + interval
            newvalue2[i] = v2 - interval
            derivative = (DFP_derivative_multifunc_multivariate(func,funcnum,newvarnumvec,newvalue1; :FuncType => FuncType) -  DFP_derivative_multifunc_multivariate(func,funcnum,newvarnumvec,newvalue2; :FuncType => FuncType))/(interval + interval)
            return DFP_setPrecision(derivative,oldprec)
        end
    end

    # FuncType = "MultiParameters" or "VectorParameters"
    function DFP_grad(func::Function,value::Vector{DFP{N}};FuncType::String="MultiParameters") where {N}
        local len = length(value)
        local grad = Vector{DFP{N}}(undef, len)
        # For each element in the vector value, we calculate the inverval
        @inbounds for i = 1:len
            grad[i] = DFP_derivative_multivariate(func,[i],value; :FuncType => FuncType)
        end
        return grad
    end

    # FuncType = "MultiParameters" or "VectorParameters"
    function DFP_jacobian(func::Function,value::Vector{DFP{N}};FuncType::String="MultiParameters") where {N}
        local len = length(value)
        local jacobian = Array{DFP{N},2}(undef, len,len)
        @inbounds for i = 1:len
            for j = 1:len
                jacobian[i,j] = DFP_derivative_multifunc_multivariate(func,i,[j],value; :FuncType => FuncType)
            end
        end
        return jacobian
    end

    # FuncType = "MultiParameters" or "VectorParameters"
    function DFP_hessian(func::Function,value::Vector{DFP{N}};FuncType::String="MultiParameters") where {N}
        local len = length(value)
        local hessian = Array{DFP{N},2}(undef, len,len)
        @inbounds for i = 1:len
            for j = 1:len
                hessian[i,j] = DFP_derivative_multivariate(func,[i,j],value; :FuncType => FuncType)
            end
        end
        return hessian
    end

    function DFP_norm(A)
        prec = DFP_getPrecision(first(A))
        total = DFP_createZero(prec)
        for x in A
            total += x * x
        end
        sqrt(total)
    end

    function Create_MathematicaLine(tracklog::Array{Any,1},func::Function;FuncType::String="MultiParameters")
        MultiParam_flag = FuncType == "MultiParameters"
        oldprec = DFP_getPrecision(tracklog[1][1])
        newprec = 4
        text = "Graphics3D[{"
        # Change newtracklog with newprec and get rid of redundant points
        newtracklog = map(x -> DFP_setPrecision.(x,newprec),tracklog)
        newtracklog = reverse(newtracklog)
        while newtracklog[1] == newtracklog[2]
            popfirst!(newtracklog)
        end
        newtracklog = reverse(newtracklog)
        # Now perform the Graphics primitive
        len = length(newtracklog)
        oldxvalue = DFP_toHardFloatingPointString(newtracklog[1][1])
        oldyvalue = DFP_toHardFloatingPointString(newtracklog[1][2])
        temp = DFP_setPrecision.(newtracklog[1],oldprec)
        oldzvalue = DFP_toHardFloatingPointString(DFP_setPrecision(MultiParam_flag ?  func(temp...) : func(temp) ,newprec))
        for n = 2:len
            if n>2
                text *= ","
            end
            text *= "{Thick, "
            text *= iseven(n) ? "Red" : "Blue"
            text *= ", Line[{"
            text *= "{$(oldxvalue),$(oldyvalue),$(oldzvalue)},"
            xvalue = DFP_toHardFloatingPointString(newtracklog[n][1])
            yvalue = DFP_toHardFloatingPointString(newtracklog[n][2])
            temp = DFP_setPrecision.(newtracklog[n],oldprec)
            zvalue = DFP_toHardFloatingPointString(DFP_setPrecision(MultiParam_flag ?  func(temp...) : func(temp) ,newprec))
            text *= "{$(xvalue),$(yvalue),$(zvalue)}"
            text *= "}]}"
            oldxvalue = xvalue
            oldyvalue = yvalue
            oldzvalue = zvalue
        end
        text *= "}]"
        return [ text ]
    end

    DFP_convertVecVectoString(array::Vector{Any}) = map(x->map(x->DFP_toCommonString(x),x),array)

    DFP_convertVecVectoFloat64(array::Vector{Any}) = map(x->map(x->DFP_toFloat64(x),x),array)




## precompile function for the DFP module

# File opened at 2024-08-05 18:04
# Inside function print_DFP(f,4)
DFP_precompile_001 = convert(DFP{4},Base.pi)
DFP_precompile_002 = sqrt(DFP{4}(2))
DFP_precompile_003 = sqrt(DFP{4}(0.7))
DFP_precompile_004 = DFP_precompile_001 + DFP_precompile_002
DFP_precompile_005 = DFP_precompile_001 - DFP_precompile_002
DFP_precompile_007 = DFP_precompile_001 * DFP_precompile_002
DFP_precompile_008 = DFP_precompile_001 / DFP_precompile_002
DFP_precompile_009 = DFP_precompile_001 ^ DFP_precompile_002
DFP_precompile_010 = sin(DFP_precompile_002)
DFP_precompile_011 = cos(DFP_precompile_002)
DFP_precompile_012 = tan(DFP_precompile_002)
DFP_precompile_013 = log(DFP_precompile_002)
DFP_precompile_014 = log(DFP_precompile_001,DFP_precompile_002)
DFP_precompile_015 = log2(DFP_precompile_002)
DFP_precompile_016 = log10(DFP_precompile_002)
DFP_precompile_017 = exp(DFP_precompile_002)
DFP_precompile_018 = exp2(DFP_precompile_002)
DFP_precompile_019 = exp10(DFP_precompile_002)
DFP_precompile_020 = sind(DFP_precompile_002)
DFP_precompile_021 = cosd(DFP_precompile_002)
DFP_precompile_022 = tand(DFP_precompile_002)
DFP_precompile_023 = asin(DFP_precompile_003)
DFP_precompile_024 = acos(DFP_precompile_003)
DFP_precompile_025 = atan(DFP_precompile_002)
DFP_precompile_026 = asind(DFP_precompile_003)
DFP_precompile_027 = acosd(DFP_precompile_003)
DFP_precompile_028 = atand(DFP_precompile_002)
DFP_precompile_029 = atan(DFP_precompile_003,DFP_precompile_002)
DFP_precompile_030 = atand(DFP_precompile_003,DFP_precompile_002)
DFP_precompile_031 = DFP_precompile_003 == DFP_precompile_002
DFP_precompile_032 = DFP_precompile_003 != DFP_precompile_002
DFP_precompile_033 = DFP_precompile_003 < DFP_precompile_002
DFP_precompile_034 = DFP_precompile_003 <= DFP_precompile_002
DFP_precompile_035 = DFP_precompile_003 > DFP_precompile_002
DFP_precompile_036 = DFP_precompile_003 >= DFP_precompile_002
# Existing function print_DFP(f,4)

# Inside function print_DFP(f,8)
DFP_precompile_037 = convert(DFP{8},Base.pi)
DFP_precompile_038 = sqrt(DFP{8}(2))
DFP_precompile_039 = sqrt(DFP{8}(0.7))
DFP_precompile_040 = DFP_precompile_037 + DFP_precompile_038
DFP_precompile_041 = DFP_precompile_037 - DFP_precompile_038
DFP_precompile_043 = DFP_precompile_037 * DFP_precompile_038
DFP_precompile_044 = DFP_precompile_037 / DFP_precompile_038
DFP_precompile_045 = DFP_precompile_037 ^ DFP_precompile_038
DFP_precompile_046 = sin(DFP_precompile_038)
DFP_precompile_047 = cos(DFP_precompile_038)
DFP_precompile_048 = tan(DFP_precompile_038)
DFP_precompile_049 = log(DFP_precompile_038)
DFP_precompile_050 = log(DFP_precompile_037,DFP_precompile_038)
DFP_precompile_051 = log2(DFP_precompile_038)
DFP_precompile_052 = log10(DFP_precompile_038)
DFP_precompile_053 = exp(DFP_precompile_038)
DFP_precompile_054 = exp2(DFP_precompile_038)
DFP_precompile_055 = exp10(DFP_precompile_038)
DFP_precompile_056 = sind(DFP_precompile_038)
DFP_precompile_057 = cosd(DFP_precompile_038)
DFP_precompile_058 = tand(DFP_precompile_038)
DFP_precompile_059 = asin(DFP_precompile_039)
DFP_precompile_060 = acos(DFP_precompile_039)
DFP_precompile_061 = atan(DFP_precompile_038)
DFP_precompile_062 = asind(DFP_precompile_039)
DFP_precompile_063 = acosd(DFP_precompile_039)
DFP_precompile_064 = atand(DFP_precompile_038)
DFP_precompile_065 = atan(DFP_precompile_039,DFP_precompile_038)
DFP_precompile_066 = atand(DFP_precompile_039,DFP_precompile_038)
DFP_precompile_067 = DFP_precompile_039 == DFP_precompile_038
DFP_precompile_068 = DFP_precompile_039 != DFP_precompile_038
DFP_precompile_069 = DFP_precompile_039 < DFP_precompile_038
DFP_precompile_070 = DFP_precompile_039 <= DFP_precompile_038
DFP_precompile_071 = DFP_precompile_039 > DFP_precompile_038
DFP_precompile_072 = DFP_precompile_039 >= DFP_precompile_038
# Existing function print_DFP(f,8)

# Inside function print_DFP(f,16)
DFP_precompile_073 = convert(DFP{16},Base.pi)
DFP_precompile_074 = sqrt(DFP{16}(2))
DFP_precompile_075 = sqrt(DFP{16}(0.7))
DFP_precompile_076 = DFP_precompile_073 + DFP_precompile_074
DFP_precompile_077 = DFP_precompile_073 - DFP_precompile_074
DFP_precompile_079 = DFP_precompile_073 * DFP_precompile_074
DFP_precompile_080 = DFP_precompile_073 / DFP_precompile_074
DFP_precompile_081 = DFP_precompile_073 ^ DFP_precompile_074
DFP_precompile_082 = sin(DFP_precompile_074)
DFP_precompile_083 = cos(DFP_precompile_074)
DFP_precompile_084 = tan(DFP_precompile_074)
DFP_precompile_085 = log(DFP_precompile_074)
DFP_precompile_086 = log(DFP_precompile_073,DFP_precompile_074)
DFP_precompile_087 = log2(DFP_precompile_074)
DFP_precompile_088 = log10(DFP_precompile_074)
DFP_precompile_089 = exp(DFP_precompile_074)
DFP_precompile_090 = exp2(DFP_precompile_074)
DFP_precompile_091 = exp10(DFP_precompile_074)
DFP_precompile_092 = sind(DFP_precompile_074)
DFP_precompile_093 = cosd(DFP_precompile_074)
DFP_precompile_094 = tand(DFP_precompile_074)
DFP_precompile_095 = asin(DFP_precompile_075)
DFP_precompile_096 = acos(DFP_precompile_075)
DFP_precompile_097 = atan(DFP_precompile_074)
DFP_precompile_098 = asind(DFP_precompile_075)
DFP_precompile_099 = acosd(DFP_precompile_075)
DFP_precompile_100 = atand(DFP_precompile_074)
DFP_precompile_101 = atan(DFP_precompile_075,DFP_precompile_074)
DFP_precompile_102 = atand(DFP_precompile_075,DFP_precompile_074)
DFP_precompile_103 = DFP_precompile_075 == DFP_precompile_074
DFP_precompile_104 = DFP_precompile_075 != DFP_precompile_074
DFP_precompile_105 = DFP_precompile_075 < DFP_precompile_074
DFP_precompile_106 = DFP_precompile_075 <= DFP_precompile_074
DFP_precompile_107 = DFP_precompile_075 > DFP_precompile_074
DFP_precompile_108 = DFP_precompile_075 >= DFP_precompile_074
# Existing function print_DFP(f,16)

# Inside function print_DFP(f,32)
DFP_precompile_109 = convert(DFP{32},Base.pi)
DFP_precompile_110 = sqrt(DFP{32}(2))
DFP_precompile_111 = sqrt(DFP{32}(0.7))
DFP_precompile_112 = DFP_precompile_109 + DFP_precompile_110
DFP_precompile_113 = DFP_precompile_109 - DFP_precompile_110
DFP_precompile_115 = DFP_precompile_109 * DFP_precompile_110
DFP_precompile_116 = DFP_precompile_109 / DFP_precompile_110
DFP_precompile_117 = DFP_precompile_109 ^ DFP_precompile_110
DFP_precompile_118 = sin(DFP_precompile_110)
DFP_precompile_119 = cos(DFP_precompile_110)
DFP_precompile_120 = tan(DFP_precompile_110)
DFP_precompile_121 = log(DFP_precompile_110)
DFP_precompile_122 = log(DFP_precompile_109,DFP_precompile_110)
DFP_precompile_123 = log2(DFP_precompile_110)
DFP_precompile_124 = log10(DFP_precompile_110)
DFP_precompile_125 = exp(DFP_precompile_110)
DFP_precompile_126 = exp2(DFP_precompile_110)
DFP_precompile_127 = exp10(DFP_precompile_110)
DFP_precompile_128 = sind(DFP_precompile_110)
DFP_precompile_129 = cosd(DFP_precompile_110)
DFP_precompile_130 = tand(DFP_precompile_110)
DFP_precompile_131 = asin(DFP_precompile_111)
DFP_precompile_132 = acos(DFP_precompile_111)
DFP_precompile_133 = atan(DFP_precompile_110)
DFP_precompile_134 = asind(DFP_precompile_111)
DFP_precompile_135 = acosd(DFP_precompile_111)
DFP_precompile_136 = atand(DFP_precompile_110)
DFP_precompile_137 = atan(DFP_precompile_111,DFP_precompile_110)
DFP_precompile_138 = atand(DFP_precompile_111,DFP_precompile_110)
DFP_precompile_139 = DFP_precompile_111 == DFP_precompile_110
DFP_precompile_140 = DFP_precompile_111 != DFP_precompile_110
DFP_precompile_141 = DFP_precompile_111 < DFP_precompile_110
DFP_precompile_142 = DFP_precompile_111 <= DFP_precompile_110
DFP_precompile_143 = DFP_precompile_111 > DFP_precompile_110
DFP_precompile_144 = DFP_precompile_111 >= DFP_precompile_110
# Existing function print_DFP(f,32)

# Inside function print_DFP(f,64)
DFP_precompile_145 = convert(DFP{64},Base.pi)
DFP_precompile_146 = sqrt(DFP{64}(2))
DFP_precompile_147 = sqrt(DFP{64}(0.7))
DFP_precompile_148 = DFP_precompile_145 + DFP_precompile_146
DFP_precompile_149 = DFP_precompile_145 - DFP_precompile_146
DFP_precompile_151 = DFP_precompile_145 * DFP_precompile_146
DFP_precompile_152 = DFP_precompile_145 / DFP_precompile_146
DFP_precompile_153 = DFP_precompile_145 ^ DFP_precompile_146
DFP_precompile_154 = sin(DFP_precompile_146)
DFP_precompile_155 = cos(DFP_precompile_146)
DFP_precompile_156 = tan(DFP_precompile_146)
DFP_precompile_157 = log(DFP_precompile_146)
DFP_precompile_158 = log(DFP_precompile_145,DFP_precompile_146)
DFP_precompile_159 = log2(DFP_precompile_146)
DFP_precompile_160 = log10(DFP_precompile_146)
DFP_precompile_161 = exp(DFP_precompile_146)
DFP_precompile_162 = exp2(DFP_precompile_146)
DFP_precompile_163 = exp10(DFP_precompile_146)
DFP_precompile_164 = sind(DFP_precompile_146)
DFP_precompile_165 = cosd(DFP_precompile_146)
DFP_precompile_166 = tand(DFP_precompile_146)
DFP_precompile_167 = asin(DFP_precompile_147)
DFP_precompile_168 = acos(DFP_precompile_147)
DFP_precompile_169 = atan(DFP_precompile_146)
DFP_precompile_170 = asind(DFP_precompile_147)
DFP_precompile_171 = acosd(DFP_precompile_147)
DFP_precompile_172 = atand(DFP_precompile_146)
DFP_precompile_173 = atan(DFP_precompile_147,DFP_precompile_146)
DFP_precompile_174 = atand(DFP_precompile_147,DFP_precompile_146)
DFP_precompile_175 = DFP_precompile_147 == DFP_precompile_146
DFP_precompile_176 = DFP_precompile_147 != DFP_precompile_146
DFP_precompile_177 = DFP_precompile_147 < DFP_precompile_146
DFP_precompile_178 = DFP_precompile_147 <= DFP_precompile_146
DFP_precompile_179 = DFP_precompile_147 > DFP_precompile_146
DFP_precompile_180 = DFP_precompile_147 >= DFP_precompile_146
# Existing function print_DFP(f,64)

# Inside function print_DFP(f,80)
DFP_precompile_181 = convert(DFP{80},Base.pi)
DFP_precompile_182 = sqrt(DFP{80}(2))
DFP_precompile_183 = sqrt(DFP{80}(0.7))
DFP_precompile_184 = DFP_precompile_181 + DFP_precompile_182
DFP_precompile_185 = DFP_precompile_181 - DFP_precompile_182
DFP_precompile_187 = DFP_precompile_181 * DFP_precompile_182
DFP_precompile_188 = DFP_precompile_181 / DFP_precompile_182
DFP_precompile_189 = DFP_precompile_181 ^ DFP_precompile_182
DFP_precompile_190 = sin(DFP_precompile_182)
DFP_precompile_191 = cos(DFP_precompile_182)
DFP_precompile_192 = tan(DFP_precompile_182)
DFP_precompile_193 = log(DFP_precompile_182)
DFP_precompile_194 = log(DFP_precompile_181,DFP_precompile_182)
DFP_precompile_195 = log2(DFP_precompile_182)
DFP_precompile_196 = log10(DFP_precompile_182)
DFP_precompile_197 = exp(DFP_precompile_182)
DFP_precompile_198 = exp2(DFP_precompile_182)
DFP_precompile_199 = exp10(DFP_precompile_182)
DFP_precompile_200 = sind(DFP_precompile_182)
DFP_precompile_201 = cosd(DFP_precompile_182)
DFP_precompile_202 = tand(DFP_precompile_182)
DFP_precompile_203 = asin(DFP_precompile_183)
DFP_precompile_204 = acos(DFP_precompile_183)
DFP_precompile_205 = atan(DFP_precompile_182)
DFP_precompile_206 = asind(DFP_precompile_183)
DFP_precompile_207 = acosd(DFP_precompile_183)
DFP_precompile_208 = atand(DFP_precompile_182)
DFP_precompile_209 = atan(DFP_precompile_183,DFP_precompile_182)
DFP_precompile_210 = atand(DFP_precompile_183,DFP_precompile_182)
DFP_precompile_211 = DFP_precompile_183 == DFP_precompile_182
DFP_precompile_212 = DFP_precompile_183 != DFP_precompile_182
DFP_precompile_213 = DFP_precompile_183 < DFP_precompile_182
DFP_precompile_214 = DFP_precompile_183 <= DFP_precompile_182
DFP_precompile_215 = DFP_precompile_183 > DFP_precompile_182
DFP_precompile_216 = DFP_precompile_183 >= DFP_precompile_182
# Existing function print_DFP(f,80)

# Inside function print_DFP(f,100)
DFP_precompile_217 = convert(DFP{100},Base.pi)
DFP_precompile_218 = sqrt(DFP{100}(2))
DFP_precompile_219 = sqrt(DFP{100}(0.7))
DFP_precompile_220 = DFP_precompile_217 + DFP_precompile_218
DFP_precompile_221 = DFP_precompile_217 - DFP_precompile_218
DFP_precompile_223 = DFP_precompile_217 * DFP_precompile_218
DFP_precompile_224 = DFP_precompile_217 / DFP_precompile_218
DFP_precompile_225 = DFP_precompile_217 ^ DFP_precompile_218
DFP_precompile_226 = sin(DFP_precompile_218)
DFP_precompile_227 = cos(DFP_precompile_218)
DFP_precompile_228 = tan(DFP_precompile_218)
DFP_precompile_229 = log(DFP_precompile_218)
DFP_precompile_230 = log(DFP_precompile_217,DFP_precompile_218)
DFP_precompile_231 = log2(DFP_precompile_218)
DFP_precompile_232 = log10(DFP_precompile_218)
DFP_precompile_233 = exp(DFP_precompile_218)
DFP_precompile_234 = exp2(DFP_precompile_218)
DFP_precompile_235 = exp10(DFP_precompile_218)
DFP_precompile_236 = sind(DFP_precompile_218)
DFP_precompile_237 = cosd(DFP_precompile_218)
DFP_precompile_238 = tand(DFP_precompile_218)
DFP_precompile_239 = asin(DFP_precompile_219)
DFP_precompile_240 = acos(DFP_precompile_219)
DFP_precompile_241 = atan(DFP_precompile_218)
DFP_precompile_242 = asind(DFP_precompile_219)
DFP_precompile_243 = acosd(DFP_precompile_219)
DFP_precompile_244 = atand(DFP_precompile_218)
DFP_precompile_245 = atan(DFP_precompile_219,DFP_precompile_218)
DFP_precompile_246 = atand(DFP_precompile_219,DFP_precompile_218)
DFP_precompile_247 = DFP_precompile_219 == DFP_precompile_218
DFP_precompile_248 = DFP_precompile_219 != DFP_precompile_218
DFP_precompile_249 = DFP_precompile_219 < DFP_precompile_218
DFP_precompile_250 = DFP_precompile_219 <= DFP_precompile_218
DFP_precompile_251 = DFP_precompile_219 > DFP_precompile_218
DFP_precompile_252 = DFP_precompile_219 >= DFP_precompile_218
# Existing function print_DFP(f,100)


end
