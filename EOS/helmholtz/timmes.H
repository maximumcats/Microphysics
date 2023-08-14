#ifndef TIMMES_H
#define TIMMES_H

#include <AMReX_REAL.H>

struct eos_timmes_t : eos_extra_t {
    Real dpdrdT{};
    Real dsdrdT{};
};

// routine dfermi gets the Fermi-Dirac functions and their derivaties
// routine fdfunc1 forms the integrand of the Fermi-Dirac functions
// routine fdfunc2 same as fdfunc but with the change of variable z**2=x
// routine dqleg020 does 20 point Gauss-Legendre integration 14 fig accuracy
// routine dqlag020 does 20 point Gauss-Laguerre integration 14 fig accuracy

// forms the fermi-dirac integrand and its first and second
// derivatives with eta and theta.

// input:
// x is the integration variable
// par[0] is the double precision index
// par[1] is the degeneravy parameter
// par[2] is the relativity parameter

// output:
// fd is the integrand
// fdeta is the first derivative with eta
// fdeta2 is the second derivative with eta
// fdtheta is the first derivative with theta
// fdtheta2 is the second derivative with theta
// fdetadtheta is the mixed second derivative

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void fdfunc1 (Real x, Real* par,
              Real& fd, Real& fdeta, Real& fdtheta,
              Real& fdeta2, Real& fdtheta2, Real& fdetadtheta)
{
    // initialize
    Real dk    = par[0];
    Real eta   = par[1];
    Real theta = par[2];
    Real xdk   = std::pow(x, dk);
    Real xst   = 1.0_rt + 0.5_rt * x * theta;
    Real dxst  = std::sqrt(xst);

    // avoid overflow in the exponentials at large x
    if ((x - eta) < 1.0e2_rt) {
        Real factor = std::exp(x - eta);
        Real denom  = factor + 1.0_rt;
        Real denomi = 1.0_rt / denom;
        fd          = xdk * dxst * denomi;
        fdeta       = fd * factor * denomi;
        fdeta2      = (2.0_rt * factor * denomi - 1.0_rt) * fdeta;
        Real denom2 = 1.0_rt / (4.0_rt * xst);
        fdtheta     = fd * x * denom2;
        fdtheta2    = -fdtheta * x * denom2;
        fdetadtheta = fdtheta * factor * denomi;
    }
    else {
        Real factor = std::exp(eta - x);
        fd          = xdk * dxst * factor;
        fdeta       = fd;
        fdeta2      = fd;
        Real denom2 = 1.0_rt / (4.0_rt * xst);
        fdtheta     = fd * x * denom2;
        fdtheta2    = -fdtheta * x * denom2;
        fdetadtheta = fdtheta;
    }
}

// Forms the Fermi-Dirac integrand and its first and second
// derivatives with eta and theta when the variable change z**2=x
// has been made.

// input:
// x is the integration variable
// par[0] is the double precision index
// par[1] is the degeneravy parameter
// par[2] is the relativity parameter

// output:
// fd is the integrand
// fdeta is the first derivative with eta
// fdeta2 is the second derivative with eta
// fdtheta is the first derivative with theta
// fdtheta2 is the second derivative with theta
// fdetadtheta is the mixed second derivative

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void fdfunc2 (Real x, Real* par,
              Real& fd, Real& fdeta, Real& fdtheta,
              Real& fdeta2, Real& fdtheta2, Real& fdetadtheta)
{
    Real dk    = par[0];
    Real eta   = par[1];
    Real theta = par[2];
    Real xsq   = x * x;
    Real xdk   = std::pow(x, 2.0_rt * dk + 1.0_rt);
    Real xst   = 1.0_rt + 0.5_rt * xsq * theta;
    Real dxst  = std::sqrt(xst);

    // avoid an overflow in the denominator at large x:
    if ((xsq - eta) < 1.0e2_rt) {
        Real factor = std::exp(xsq - eta);
        Real denom  = factor + 1.0_rt;
        Real denomi = 1.0_rt / denom;
        fd          = 2.0_rt * xdk * dxst * denomi;
        fdeta       = fd * factor * denomi;
        fdeta2      = (2.0_rt * factor * denomi - 1.0_rt) * fdeta;
        Real denom2 = 1.0_rt / (4.0_rt * xst);
        fdtheta     = fd * xsq * denom2;
        fdtheta2    = -fdtheta * xsq * denom2;
        fdetadtheta = fdtheta * factor * denomi;
    }
    else {
        Real factor = std::exp(eta - xsq);
        fd          = 2.0_rt * xdk * dxst * factor;
        fdeta       = fd;
        fdeta2      = fd;
        Real denom2 = 1.0_rt / (4.0_rt * xst);
        fdtheta     = fd * xsq * denom2;
        fdtheta2    = -fdtheta * xsq * denom2;
        fdetadtheta = fdtheta;
    }
}

// 20 point gauss-legendre rule for the Fermi-Dirac function and
// its derivatives with respect to eta and theta.
// On input integrand determines which function contains the integrand,
// a is the lower end point of the interval, b is the higher end point,
// par is an array of constant parameters to be passed to subroutine f,
// and n is the length of the par array. On output result is the
// approximation from applying the 20-point gauss-legendre rule,
// drdeta is the derivative with respect to eta, and drdtheta is the
// derivative with respect to theta.
//
// Note: since the number of nodes is even, zero is not an abscissa.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dqleg020 (int integrand, Real a, Real b,
               Real& result, Real& drdeta, Real& drdtheta,
               Real& drdeta2, Real& drdtheta2, Real& drdetadtheta,
               Real* par)
{
    // the abscissae and weights are given for the interval (-1,1).
    // xg     - abscissae of the 20-point gauss-legendre rule
    //          for half of the usual run (-1,1), i.e.
    //          the positive nodes of the 20-point rule
    // wg     - weights of the 20-point gauss rule.
    //
    // abscissae and weights were evaluated with 100 decimal digit arithmetic.

    Array1D<Real, 1, 10> xg = {7.65265211334973337546404093988382110e-2_rt,
                               2.27785851141645078080496195368574624e-1_rt,
                               3.73706088715419560672548177024927237e-1_rt,
                               5.10867001950827098004364050955250998e-1_rt,
                               6.36053680726515025452836696226285936e-1_rt,
                               7.46331906460150792614305070355641590e-1_rt,
                               8.39116971822218823394529061701520685e-1_rt,
                               9.12234428251325905867752441203298113e-1_rt,
                               9.63971927277913791267666131197277221e-1_rt,
                               9.93128599185094924786122388471320278e-1_rt};

    Array1D<Real, 1, 10> wg = {1.52753387130725850698084331955097593e-1_rt,
                               1.49172986472603746787828737001969436e-1_rt,
                               1.42096109318382051329298325067164933e-1_rt,
                               1.31688638449176626898494499748163134e-1_rt,
                               1.18194531961518417312377377711382287e-1_rt,
                               1.01930119817240435036750135480349876e-1_rt,
                               8.32767415767047487247581432220462061e-2_rt,
                               6.26720483341090635695065351870416063e-2_rt,
                               4.06014298003869413310399522749321098e-2_rt,
                               1.76140071391521183118619623518528163e-2_rt};

    //           list of major variables
    //           -----------------------
    //
    //           absc   - abscissa
    //           fval*  - function value
    //           result - result of the 20-point gauss formula

    Real center       = 0.5_rt * (a + b);
    Real hlfrun       = 0.5_rt * (b - a);
    result       = 0.0_rt;
    drdeta       = 0.0_rt;
    drdtheta     = 0.0_rt;
    drdeta2      = 0.0_rt;
    drdtheta2    = 0.0_rt;
    drdetadtheta = 0.0_rt;
    for (int j = 1; j <= 10; ++j) {
        Real absc1 = center + hlfrun * xg(j);
        Real absc2 = center - hlfrun * xg(j);

        Array1D<Real, 1, 2> fval, deta, dtheta, deta2, dtheta2, detadtheta;

        if (integrand == 1) {
            fdfunc1(absc1, par, fval(1), deta(1), dtheta(1),
                    deta2(1), dtheta2(1), detadtheta(1));
            fdfunc1(absc2, par, fval(2), deta(2), dtheta(2),
                    deta2(2), dtheta2(2), detadtheta(2));
        }
        else {
            fdfunc2(absc1, par, fval(1), deta(1), dtheta(1),
                    deta2(1), dtheta2(1), detadtheta(1));
            fdfunc2(absc2, par, fval(2), deta(2), dtheta(2),
                    deta2(2), dtheta2(2), detadtheta(2));
        }

        result       = result    + (fval(1)    + fval(2)) * wg(j);
        drdeta       = drdeta    + (deta(1)    + deta(2)) * wg(j);
        drdtheta     = drdtheta  + (dtheta(1)  + dtheta(2)) * wg(j);
        drdeta2      = drdeta2   + (deta2(1)   + deta2(2)) * wg(j);
        drdtheta2    = drdtheta2 + (dtheta2(1) + dtheta2(2)) * wg(j);
        drdetadtheta = drdetadtheta + (detadtheta(1) + detadtheta(2)) * wg(j);
    }

    result       = result * hlfrun;
    drdeta       = drdeta * hlfrun;
    drdtheta     = drdtheta * hlfrun;
    drdeta2      = drdeta2 * hlfrun;
    drdtheta2    = drdtheta2 * hlfrun;
    drdetadtheta = drdetadtheta * hlfrun;
}

// 20 point Gauss-Laguerre rule for the Fermi-Dirac function.
// On input integrand determines which function to evaluate,
// f(x) = g(x) * w(x), where w(x) is the gaussian weight,
// w(x) = std::exp(-(x - a) / b) and g(x) a smooth function,
// a is the lower end point of the interval, b is the higher end point,
// par is an array of constant parameters to be passed to the function f,
// and n is the length of the par array. On output result is the
// approximation from applying the 20-point Gauss-Laguerre rule.
// Since the number of nodes is even, zero is not an abscissa.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dqlag020 (int integrand, Real a, Real b,
               Real& result, Real& drdeta, Real& drdtheta,
               Real& drdeta2, Real& drdtheta2, Real& drdetadtheta,
               Real* par)
{
    // the abscissae and weights are given for the interval (0,+inf).
    // xg     - abscissae of the 20-point gauss-laguerre rule
    // wg     - weights of the 20-point gauss rule. since f yet
    //          includes the weight function, the values in wg
    //          are actually exp(xg) times the standard
    //          gauss-laguerre weights
    //
    // abscissae and weights were evaluated with 100 decimal digit arithmetic.

    Array1D<Real, 1, 20> xg = {7.05398896919887533666890045842150958e-2_rt,
                               3.72126818001611443794241388761146636e-1_rt,
                                9.16582102483273564667716277074183187e-1_rt,
                                1.70730653102834388068768966741305070_rt,
                                2.74919925530943212964503046049481338_rt,
                                4.04892531385088692237495336913333219_rt,
                                5.61517497086161651410453988565189234_rt,
                                7.45901745367106330976886021837181759_rt,
                                9.59439286958109677247367273428279837_rt,
                                1.20388025469643163096234092988655158e1_rt,
                                1.48142934426307399785126797100479756e1_rt,
                                1.79488955205193760173657909926125096e1_rt,
                                2.14787882402850109757351703695946692e1_rt,
                                2.54517027931869055035186774846415418e1_rt,
                                2.99325546317006120067136561351658232e1_rt,
                                3.50134342404790000062849359066881395e1_rt,
                                4.08330570567285710620295677078075526e1_rt,
                                4.76199940473465021399416271528511211e1_rt,
                                5.58107957500638988907507734444972356e1_rt,
                                6.65244165256157538186403187914606659e1_rt};

    Array1D<Real, 1, 20> wg = {1.81080062418989255451675405913110644e-1_rt,
                               4.22556767878563974520344172566458197e-1_rt,
                               6.66909546701848150373482114992515927e-1_rt,
                               9.15352372783073672670604684771868067e-1_rt,
                               1.16953970719554597380147822239577476_rt,
                               1.43135498592820598636844994891514331_rt,
                               1.70298113798502272402533261633206720_rt,
                               1.98701589079274721410921839275129020_rt,
                               2.28663578125343078546222854681495651_rt,
                               2.60583472755383333269498950954033323_rt,
                               2.94978373421395086600235416827285951_rt,
                               3.32539578200931955236951937421751118_rt,
                               3.74225547058981092111707293265377811_rt,
                               4.21423671025188041986808063782478746_rt,
                               4.76251846149020929695292197839096371_rt,
                               5.42172604424557430380308297989981779_rt,
                               6.25401235693242129289518490300707542_rt,
                               7.38731438905443455194030019196464791_rt,
                               9.15132873098747960794348242552950528_rt,
                               1.28933886459399966710262871287485278e1_rt};

    //           list of major variables
    //           -----------------------
    //           absc   - abscissa
    //           fval*  - function value
    //           result - result of the 20-point gauss formula

    result       = 0.0_rt;
    drdeta       = 0.0_rt;
    drdtheta     = 0.0_rt;
    drdeta2      = 0.0_rt;
    drdtheta2    = 0.0_rt;
    drdetadtheta = 0.0_rt;

    for (int j = 1; j <= 20; ++j) {
        Real absc = a + b * xg(j);

        Real fval, deta, dtheta, deta2, dtheta2, detadtheta;

        if (integrand == 1) {
            fdfunc1(absc, par, fval, deta, dtheta,
                    deta2, dtheta2, detadtheta);
        }
        else {
            fdfunc2(absc, par, fval, deta, dtheta,
                    deta2, dtheta2, detadtheta);            
        }

        result       = result    + fval * wg(j);
        drdeta       = drdeta    + deta * wg(j);
        drdtheta     = drdtheta  + dtheta * wg(j);
        drdeta2      = drdeta2   + deta2 * wg(j);
        drdtheta2    = drdtheta2 + dtheta2 * wg(j);
        drdetadtheta = drdetadtheta + detadtheta * wg(j);
    }

    result       = result * b;
    drdeta       = drdeta * b;
    drdtheta     = drdtheta * b;
    drdeta2      = drdeta2 * b;
    drdtheta2    = drdtheta2 * b;
    drdetadtheta = drdetadtheta * b;
}


// This routine computes the Fermi-Dirac integrals f(dk,eta,theta) of
// index dk, with degeneracy parameter eta and relativity parameter theta.
// The input is dk, the double precision index of the Fermi-Dirac function;
// eta, the degeneracy parameter; and theta, the relativity parameter.

// The output is fd, computed by applying three 10-point gauss-legendre
// and one 10-point gauss-laguerre rules over four appropriate subintervals.
// fdeta is the derivative of f with respect to eta, and
// fdthetha is the derivative of f with theta.

// reference: j.m. aparicio, apjs 117, 632 1998

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void dfermi(Real dk, Real eta, Real theta,
            Real& fd, Real& fdeta, Real& fdtheta,
            Real& fdeta2, Real& fdtheta2, Real& fdetadtheta)
{
    // Parameters defining the location of the breakpoints for the
    // subintervals of integration:
    const Real d = 3.3609_rt;
    const Real sg = 9.1186e-2_rt;
    const Real a1 = 6.7774_rt;
    const Real b1 = 1.1418_rt;
    const Real c1 = 2.9826_rt;
    const Real a2 = 3.7601_rt;
    const Real b2 = 9.3719e-2_rt;
    const Real c2 = 2.1063e-2_rt;
    const Real d2 = 3.1084e1_rt;
    const Real e2 = 1.0056_rt;
    const Real a3 = 7.5669_rt;
    const Real b3 = 1.1695_rt;
    const Real c3 = 7.5416e-1_rt;
    const Real d3 = 6.6558_rt;
    const Real e3 = -1.2819e-1_rt;

    // integrand parameters:
    Real par[3] = {dk, eta, theta};

    // definition of xi:
    Real eta1 = sg * (eta - d);
    Real xi;
    if (eta1 <= 5.0e1_rt) {
        xi = std::log(1.0_rt + std::exp(eta1)) / sg;
    }
    else {
        xi = eta - d;
    }
    Real xi2 = xi * xi;

    // definition of the x_i:
    Real x1 = (a1 + b1 * xi + c1 * xi2) /
              (1.0_rt + c1 * xi);
    Real x2 = (a2 +b2 * xi + c2 * d2 * xi2) /
              (1.0_rt + e2 * xi + c2 * xi2);
    Real x3 = (a3 + b3 * xi + c3 * d3 * xi2) /
              (1.0_rt + e3 * xi + c3 * xi2);

    // breakpoints:
    Real s1 = x1 - x2;
    Real s2 = x1;
    Real s3 = x1 + x3;
    Real s12 = std::sqrt(s1);

    // quadrature integrations:

    // 14 significant figure accuracy
    Array1D<Real, 1, 4> res, drde, drdt, drde2, drdt2, drdet;
    dqleg020(2, 0.0_rt,  s12, res(1), drde(1), drdt(1),
             drde2(1), drdt2(1), drdet(1), par);
    dqleg020(1, s1, s2, res(2), drde(2), drdt(2),
             drde2(2), drdt2(2), drdet(2), par);
    dqleg020(1, s2, s3, res(3), drde(3), drdt(3),
             drde2(3), drdt2(3), drdet(3), par);
    dqlag020(1, s3, 1.0_rt, res(4), drde(4), drdt(4),
             drde2(4), drdt2(4), drdet(4), par);

    // sum the contributions
    fd          = res(1)   + res(2)   + res(3)   + res(4);
    fdeta       = drde(1)  + drde(2)  + drde(3)  + drde(4);
    fdtheta     = drdt(1)  + drdt(2)  + drdt(3)  + drdt(4);
    fdeta2      = drde2(1) + drde2(2) + drde2(3) + drde2(4);
    fdtheta2    = drdt2(1) + drdt2(2) + drdt2(3) + drdt2(4);
    fdetadtheta = drdet(1) + drdet(2) + drdet(3) + drdet(4);
}

// Make a guess for the electron degeneracy parameter eta.
// Input is the ion number density xni, average charge zbar,
// and temperature temp. Output is a guess at the electron
// chemical potential eta.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real etages (Real xni, Real zbar, Real temp)
{
    Real eta;

    const Real pi = 3.1415926535897932384_rt;
    const Real h = 6.6260689633e-27_rt;
    const Real clight = 2.99792458e10_rt;
    const Real kerg = 1.380650424e-16_rt;
    const Real me = 9.1093821545e-28_rt;

    const Real rt2 = 1.4142135623730951_rt;
    const Real rt3 = 1.7320508075688772_rt;
    const Real rtpi = 1.7724538509055159_rt;
    const Real cpf0 = h / (me * clight);
    const Real cpf1 = 3.0_rt / (8.0_rt * pi) * cpf0 * cpf0 * cpf0;
    const Real cpf2 = 4.0_rt / cpf1;
    const Real cpf3 = 2.0_rt * rt3 * rtpi / (rt2 * cpf1);
    const Real twoth = 2.0_rt / 3.0_rt;
    const Real fa0 = 64.0_rt / (9.0_rt * pi);
    const Real forpi = 4.0_rt * pi;
    const Real mecc = me * clight * clight;

    Real xne = xni * zbar;
    Real kt = kerg * temp;
    Real beta = kt / mecc;

    // Number density of ionized electrons (c&g 24.354k) and number density at
    // turning point (c&g 24.354i). If either of these exceed the number density
    // as given by a saha equation, then pairs are important. Set eta = -1/2.

    Real x;
    if (beta >= 1.0_rt) {
        x = cpf2 * beta * beta;
    }
    else {
        x = cpf3 * beta * (1.0_rt + 0.75_rt * beta) * std::exp(-1.0_rt / beta);
    }

    if (x >= xne) {
        eta = -0.5_rt;
    }
    else {
        // Get the dimensionless number density (c&g 24.313), if it is large apply the
        // formula (c&g 24.309) to get a possible alfa, if not large do a two term
        // binomial expansion on (c&g 24.309) to estimate eta.

        Real z = std::pow(xne * cpf1, twoth);

        Real y;
        if (z >= 1.0e-6_rt) {
            y = (std::sqrt(z + 1.0_rt) - 1.0_rt) / beta;
        }
        else {
            y = z * (1.0_rt - z * 0.25_rt) * 0.5_rt / beta;
        }

        // Isolate the constant in front of the number density integral. If it is
        // small enough run the divine approximation backwards with c&g 24.43. Then
        // join it smoothly with the lower limit.

        x = std::log10(std::pow(xne, 0.6_rt) / temp);

        if (x <= 9.5_rt) {
            z = ((1.0_rt + fa0 * beta) * std::sqrt(1.0_rt + fa0 * beta * 0.5) - 1.0_rt) / fa0;
            Real tmkt = 2.0_rt * me / h * kt / h;
            Real xnefac = forpi * tmkt * std::sqrt(tmkt);
            eta = -std::log(xnefac * rtpi * (0.5_rt + 0.75_rt * z) / xne);

            if (x >= 8.5_rt) {
                eta = eta * (9.5_rt - x) + y * (1.0_rt - (9.5_rt - x));
            }
        }
        else {
            eta = y;
        }
    }

    return eta;
}

// This routine is called by a root finder to find the degeneracy parameter aa
// where the number density from a saha equation equals the number density as
// computed by the Fermi-Dirac integrals. Input mode 0 means do a root find on
// the electron number density; input mode 1 means do the full calculation.
// The output is f (the value of the function to be zeroed, xne - xnefer + xnepos)
// and df (the derivative with respect to aa of a).

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void xneroot (int mode, T& state, Real aa, Real& f, Real& df)
{
    const Real pi = 3.1415926535897932384_rt;
    const Real h = 6.6260689633e-27_rt;
    const Real me = 9.1093821545e-28_rt;
    const Real clight = 2.99792458e10_rt;
    const Real xconst = 8.0_rt * pi * std::sqrt(2.0_rt) * std::pow(me * clight / h, 3);
    const Real pconst = xconst * 2.0_rt / 3.0_rt * me * clight * clight;
    const Real econst = xconst * me * clight * clight;
    const Real mecc = me * clight * clight;
    const Real avo = 6.0221417930e23_rt;
    const Real kerg = 1.380650424e-16_rt;
    const Real dbetadt = kerg / mecc;
    const Real safe = 0.005_rt;
    const Real positron_start = 0.02_rt;

    // some common factors
    Real ytot1   = 1.0_rt / state.abar;
    Real deni    = 1.0_rt / state.rho;
    Real kt      = kerg * state.T;
    Real kti     = 1.0_rt / kt;
    Real beta    = kt / mecc;
    Real beta12  = std::sqrt(beta);
    Real beta32  = beta * beta12;
    Real beta52  = beta * beta32;
    Real etaele  = aa;

    // ion number density in 1/cm**3
    Real xni     = avo * ytot1 * state.rho;
    Real dxnidd  = avo * ytot1;
    Real dxnidt  = 0.0_rt;
    Real dxnida  = -xni * ytot1;
    Real dxnidz  = 0.0_rt;

    Real dxniddd = 0.0_rt;
    Real dxniddt = 0.0_rt;
    Real dxnidda = -dxnidd * ytot1;
    Real dxniddz = 0.0_rt;
    Real dxnidtt = 0.0_rt;
    Real dxnidta = 0.0_rt;
    Real dxnidtz = 0.0_rt;
    Real dxnidaa = -2.0_rt * dxnida * ytot1;
    Real dxnidaz = 0.0_rt;
    Real dxnidzz = 0.0_rt;

    // get the number density of free electrons
    // saha is the ratio of the ground state to the ionized state
    // this model is exact for a pure hydrogen composition
    // denion is a crude pressure ionization model

    // assume fully ionized
    Real chi        = 0.0_rt;
    Real chifac     = 0.0_rt;
    Real dchifacdt  = 0.0_rt;
    Real dchifacdz  = 0.0_rt;
    Real dchifacdtt = 0.0_rt;
    Real dchifacdtz = 0.0_rt;
    Real dchifacdzz = 0.0_rt;
    Real saha       = 0.0_rt;
    Real dsaha_dd   = 0.0_rt;
    Real dsaha_dt   = 0.0_rt;
    Real dsaha_da   = 0.0_rt;
    Real dsaha_dz   = 0.0_rt;
    Real dsaha_deta = 0.0_rt;
    Real dsaha_ddd  = 0.0_rt;
    Real dsaha_ddt  = 0.0_rt;
    Real dsaha_dda  = 0.0_rt;
    Real dsaha_ddz  = 0.0_rt;
    Real dsaha_dtt  = 0.0_rt;
    Real dsaha_dta  = 0.0_rt;
    Real dsaha_dtz  = 0.0_rt;
    Real dsaha_daa  = 0.0_rt;
    Real dsaha_daz  = 0.0_rt;
    Real dsaha_dzz  = 0.0_rt;
    Real dsaha_deta_dd = 0.0_rt;
    Real dsaha_deta_dt = 0.0_rt;
    Real dsaha_deta_da = 0.0_rt;
    Real dsaha_deta_dz = 0.0_rt;
    Real dsaha_deta2   = 0.0_rt;

    // saha factor, the fraction ionized
    Real sfac = 1.0_rt / (1.0_rt + saha);

    // first derivatives
    Real y          = -sfac * sfac;
    Real dsfac_dd   = y * dsaha_dd;
    Real dsfac_dt   = y * dsaha_dt;
    Real dsfac_da   = y * dsaha_da;
    Real dsfac_dz   = y * dsaha_dz;
    Real dsfac_deta = y * dsaha_deta;

    // second derivatives
    Real ww         = -2.0_rt * sfac;
    Real dsfac_ddd  = ww * dsfac_dd * dsaha_dd + y * dsaha_ddd;
    Real dsfac_ddt  = ww * dsfac_dt * dsaha_dd + y * dsaha_ddt;
    Real dsfac_dda  = ww * dsfac_da * dsaha_dd + y * dsaha_dda;
    Real dsfac_ddz  = ww * dsfac_dz * dsaha_dd + y * dsaha_ddz;
    Real dsfac_dtt  = ww * dsfac_dt * dsaha_dt + y * dsaha_dtt;
    Real dsfac_dta  = ww * dsfac_da * dsaha_dt + y * dsaha_dta;
    Real dsfac_dtz  = ww * dsfac_dz * dsaha_dt + y * dsaha_dtz;
    Real dsfac_daa  = ww * dsfac_da * dsaha_da + y * dsaha_daa;
    Real dsfac_daz  = ww * dsfac_dz * dsaha_da + y * dsaha_daz;
    Real dsfac_dzz  = ww * dsfac_dz * dsaha_dz + y * dsaha_dzz;
    Real dsfac_deta_dd = ww * dsfac_dd * dsaha_deta + y * dsaha_deta_dd;
    Real dsfac_deta_dt = ww * dsfac_dt * dsaha_deta + y * dsaha_deta_dt;
    Real dsfac_deta_da = ww * dsfac_da * dsaha_deta + y * dsaha_deta_da;
    Real dsfac_deta_dz = ww * dsfac_dz * dsaha_deta + y * dsaha_deta_dz;
    Real dsfac_deta2   = ww * dsfac_deta * dsaha_deta + y * dsaha_deta2;

    // effective charge
    Real zeff = state.zbar * sfac;

    // first derivatives
    Real dzeff_dd   = state.zbar * dsfac_dd;
    Real dzeff_dt   = state.zbar * dsfac_dt;
    Real dzeff_da   = state.zbar * dsfac_da;
    Real dzeff_dz   = sfac;
    Real dzeff_deta = state.zbar * dsfac_deta;

    // second derivatives
    Real dzeff_ddd  = state.zbar * dsfac_ddd;
    Real dzeff_ddt  = state.zbar * dsfac_ddt;
    Real dzeff_dda  = state.zbar * dsfac_dda;
    Real dzeff_ddz  = state.zbar * dsfac_ddz;
    Real dzeff_dtt  = state.zbar * dsfac_dtt;
    Real dzeff_dta  = state.zbar * dsfac_dta;
    Real dzeff_dtz  = state.zbar * dsfac_dtz;
    Real dzeff_daa  = state.zbar * dsfac_daa;
    Real dzeff_daz  = state.zbar * dsfac_daz;
    Real dzeff_dzz  = dsfac_dz;
    Real dzeff_deta_dd = state.zbar * dsfac_deta_dd;
    Real dzeff_deta_dt = state.zbar * dsfac_deta_dt;
    Real dzeff_deta_da = state.zbar * dsfac_deta_da;
    Real dzeff_deta_dz = dsfac_deta + state.zbar * dsfac_deta_dz;
    Real dzeff_deta2   = state.zbar * dsfac_deta2;

    // number density of free electrons
    Real xne = xni * zeff;

    // first derivatives
    Real dxne_dd    = dxnidd * zeff + xni * dzeff_dd;
    Real dxne_dt    = dxnidt * zeff + xni * dzeff_dt;
    Real dxne_da    = dxnida * zeff + xni * dzeff_da;
    Real dxne_dz    = dxnidz * zeff + xni * dzeff_dz;
    Real dxne_deta  = xni * dzeff_deta;

    // second derivatives
    Real dxne_ddd = dxniddd * zeff + 2.0_rt * dxnidd * dzeff_dd + xni * dzeff_ddd;
    Real dxne_ddt = dxniddt * zeff + dxnidd * dzeff_dt + dxnidt * dzeff_dt + xni * dzeff_ddt;
    Real dxne_dda = dxnidda * zeff + dxnidd * dzeff_da + dxnida * dzeff_dt + xni * dzeff_dda;
    Real dxne_ddz = dxniddz * zeff + dxnidd * dzeff_dz + dxnidz * dzeff_dt + xni * dzeff_ddz;
    Real dxne_dtt = dxnidtt * zeff + 2.0_rt * dxnidt * dzeff_dt + xni * dzeff_dtt;
    Real dxne_dta = dxnidta * zeff + dxnidt * dzeff_da + dxnida * dzeff_dt + xni * dzeff_dta;
    Real dxne_dtz = dxnidtz * zeff + dxnidt * dzeff_dz + dxnidz * dzeff_dt + xni * dzeff_dtz;
    Real dxne_daa = dxnidaa * zeff + 2.0_rt * dxnida * dzeff_da + xni * dzeff_daa;
    Real dxne_daz = dxnidaz * zeff + dxnida * dzeff_dz + dxnidz * dzeff_da + xni * dzeff_daz;
    Real dxne_dzz = dxnidzz * zeff + 2.0_rt * dxnidz * dzeff_dz + xni * dzeff_dzz;
    Real dxne_deta_dd = dxnidd * dzeff_deta + xni * dzeff_deta_dd;
    Real dxne_deta_dt = dxnidt * dzeff_deta + xni * dzeff_deta_dt;
    Real dxne_deta_dz = dxnidz * dzeff_deta + xni * dzeff_deta_dz;
    Real dxne_deta_da = dxnida * dzeff_deta + xni * dzeff_deta_da;
    Real dxne_deta2   = xni * dzeff_deta2;

    // get the fermi-dirac integral electron contribution
    Real f12, f12eta, f12beta, f12eta2, f12beta2, f12etabeta;
    dfermi(0.5_rt, etaele, beta, f12, f12eta, f12beta, f12eta2, f12beta2, f12etabeta);
    Real f32, f32eta, f32beta, f32eta2, f32beta2, f32etabeta;
    dfermi(1.5_rt, etaele, beta, f32, f32eta, f32beta, f32eta2, f32beta2, f32etabeta);

    Real zz   = xconst * beta32;
    ww   = xconst * 1.5_rt * beta12;
    Real yy   = f12 + beta * f32;
    Real dum1 = f12eta + beta * f32eta;
    Real dum2 = f12beta + f32 + beta * f32beta;

    Real xnefer             = zz * yy;
    Real dxnefer_deta       = zz * dum1;
    Real dxnefer_dbeta      = ww * yy + zz * dum2;
    Real dxnefer_deta2      = zz * (f12eta2 + beta * f32eta2);
    Real dxnefer_dbeta2     = 0.5_rt * ww / beta * yy + 2.0_rt * ww * dum2 +
                              zz * (f12beta2 + 2.0_rt * f32beta + beta * f32beta2);
    Real dxnefer_deta_dbeta = ww * dum1 +
                              zz * (f12etabeta + f32eta + beta * f32etabeta);

    // if the temperature is not too low, get the positron contributions
    // chemical equilibrium means etaele + etapos = eta_photon = 0.
    Real etapos             = 0.0_rt;
    Real detap_deta         = 0.0_rt;
    Real detap_dbeta        = 0.0_rt;
    Real detap_deta2        = 0.0_rt;
    Real detap_dbeta2       = 0.0_rt;
    Real detap_deta_dbeta   = 0.0_rt;
    Real xnpfer             = 0.0_rt;
    Real dxnpfer_deta       = 0.0_rt;
    Real dxnpfer_dbeta      = 0.0_rt;
    Real dxnpfer_deta2      = 0.0_rt;
    Real dxnpfer_dbeta2     = 0.0_rt;
    Real dxnpfer_deta_dbeta = 0.0_rt;

    if (beta > positron_start) {
        etapos           = -aa - 2.0_rt / beta;
        detap_deta       = -1.0_rt;
        detap_deta2      = 0.0_rt;
        detap_dbeta      = 2.0_rt / (beta * beta);
        detap_dbeta2     = -4.0_rt / (beta * beta * beta);
        detap_deta_dbeta = 0.0_rt;

        dfermi(0.5_rt, etapos, beta, f12, f12eta, f12beta, f12eta2, f12beta2, f12etabeta);
        dfermi(1.5_rt, etapos, beta, f32, f32eta, f32beta, f32eta2, f32beta2, f32etabeta);

        zz   = xconst * beta32;
        ww   = xconst * 1.5_rt * beta12;
        yy   = f12 + beta * f32;
        dum1 = f12eta + beta * f32eta;
        dum2 = f12beta + f32 + beta * f32beta;

        xnpfer         = zz * yy;
        Real dxnpfer_detap  = zz * dum1;
        dxnpfer_dbeta  = ww * yy + zz * dum2;
        Real dxnpfer_detap2 = zz * (f12eta2 + beta * f32eta2);
        dxnpfer_dbeta2 = 0.5_rt * ww / beta * yy + 2.0_rt * ww * dum2 +
                         zz * (f12beta2 + 2.0_rt * f32beta + beta * f32beta2);
        Real dxnpfer_detap_dbeta = ww * dum1 +
                                   zz * (f12etabeta + f32eta + beta * f32etabeta);

        // convert the etap derivatives to eta derivatives
        // all derived from the operator dxp = dxp/detap detap + dxp/dbeta dbeta

        dxnpfer_deta  = dxnpfer_detap * detap_deta;
        dxnpfer_dbeta = dxnpfer_dbeta + dxnpfer_detap * detap_dbeta;
        dxnpfer_deta2 = dxnpfer_detap2 * detap_deta * detap_deta +
                        dxnpfer_detap * detap_deta2;
        dxnpfer_dbeta2 = dxnpfer_dbeta2 +
                         2.0_rt * dxnpfer_detap_dbeta * detap_dbeta +
                         dxnpfer_detap2 * detap_dbeta * detap_dbeta +
                         dxnpfer_detap * detap_dbeta2;
        dxnpfer_deta_dbeta = dxnpfer_detap2 * detap_dbeta * detap_deta +
                             dxnpfer_detap_dbeta * detap_deta +
                             dxnpfer_detap * detap_deta_dbeta;
    }

    // charge neutrality means ne_ionization = ne_electrons - ne_positrons
    f  = xnefer - xnpfer - xne;

    // derivative of f with eta for newton-like root finders
    df = dxnefer_deta  - dxnpfer_deta  - dxne_deta;

    // if we are in root finder mode, return

    if (mode == 0) {
        return;
    }

    // if we are not in root finder mode, polish off the calculation

    // all the derivatives are in terms of eta and beta.
    // we want to convert to temperature, density, abar and zbar derivatives.
    // so, after the root find above on eta we have xne = xnefer - xnpfer.
    // taking the derivative of this for property p
    // dxne/deta deta + dxne/dp dp = dxnefer/deta deta + dxnefer/dbeta dbeta
    // solving for the unknown eta derivative yields
    // deta/dp = (dxne_dp - dxnefer/dbeta dbeta/dp) / (dxnefer/deta - dxne/deta)

    Real dxep_deta       = dxnefer_deta  - dxnpfer_deta;
    Real dxep_dbeta      = dxnefer_dbeta - dxnpfer_dbeta;
    Real dxep_deta2      = dxnefer_deta2  - dxnpfer_deta2;
    Real dxep_dbeta2     = dxnefer_dbeta2 - dxnpfer_dbeta2;
    Real dxep_deta_dbeta = dxnefer_deta_dbeta  - dxnpfer_deta_dbeta;

    y = 1.0_rt / (dxep_deta - dxne_deta);

    // the all important first derivatives of eta
    Real detadd = dxne_dd * y;
    Real detadt = (dxne_dt - dxep_dbeta * dbetadt) * y;
    Real detada = dxne_da * y;
    Real detadz = dxne_dz * y;

    // second derivatives
    Real detaddd = dxne_ddd * y - detadd * y * (dxep_deta2 - dxne_deta2) * detadd;
    Real detaddt = dxne_ddt * y - detadd * y * (dxep_deta2 * detadt +
                                                dxep_deta_dbeta * dbetadt - dxne_deta2 * detadt);
    Real detadda = dxne_dda * y - detadd * y * (dxep_deta2 - dxne_deta2) * detada;
    Real detaddz = dxne_ddz * y - detadd * y * (dxep_deta2 - dxne_deta2) * detadz;

    Real detadtt = (dxne_dtt - (dxep_deta_dbeta * detadt +
                                dxep_dbeta2 * dbetadt) * dbetadt) * y -
                   detadt * y * (dxep_deta2 * detadt +
                                 dxep_deta_dbeta * dbetadt - dxne_deta2 * detadt);

    Real detadta = (dxne_dta - dxep_deta_dbeta * detada * dbetadt) * y -
                   detadt * y * (dxep_deta2 * detada - dxne_deta2 * detada);

    Real detadtz = (dxne_dtz - dxep_deta_dbeta * detadz * dbetadt) * y -
                   detadt * y * (dxep_deta2 * detadz - dxne_deta2 * detadz);

    Real detadaa = dxne_daa * y - detada * y * (dxep_deta2 - dxne_deta2) * detada;
    Real detadaz = dxne_daz * y - detada * y * (dxep_deta2 - dxne_deta2) * detadz;
    Real detadzz = dxne_dzz * y - detadz * y * (dxep_deta2 - dxne_deta2) * detadz;

    // first derivatives of the effective charge
    Real dzeffdd = dzeff_deta * detadd + dzeff_dd;
    Real dzeffdt = dzeff_deta * detadt + dzeff_dt;
    Real dzeffda = dzeff_deta * detada + dzeff_da;
    Real dzeffdz = dzeff_deta * detadz + dzeff_dz;

    // second derivatives
    Real dzeffddd = dzeff_deta_dd * detadd + dzeff_deta * detaddd + dzeff_ddd;
    Real dzeffddt = dzeff_deta_dt * detadd + dzeff_deta * detaddt + dzeff_ddt;
    Real dzeffdda = dzeff_deta_da * detadd + dzeff_deta * detadda + dzeff_dda;
    Real dzeffddz = dzeff_deta_dz * detadd + dzeff_deta * detaddz + dzeff_ddz;
    Real dzeffdtt = dzeff_deta_dt * detadt + dzeff_deta * detadtt + dzeff_dtt;
    Real dzeffdta = dzeff_deta_da * detadt + dzeff_deta * detadta + dzeff_dta;
    Real dzeffdtz = dzeff_deta_dz * detadt + dzeff_deta * detadtz + dzeff_dtz;
    Real dzeffdaa = dzeff_deta_da * detada + dzeff_deta * detadaa + dzeff_daa;
    Real dzeffdaz = dzeff_deta_dz * detada + dzeff_deta * detadaz + dzeff_daz;
    Real dzeffdzz = dzeff_deta_dz * detadz + dzeff_deta * detadzz + dzeff_dzz;

    // first derivatives of the electron number density
    Real dxnedd = dxnidd  *  zeff + xni  *  dzeffdd;
    Real dxnedt = dxnidt  *  zeff + xni  *  dzeffdt;
    Real dxneda = dxnida  *  zeff + xni  *  dzeffda;
    Real dxnedz = dxnidz  *  zeff + xni  *  dzeffdz;

    // second derivatives
    Real dxneddd = dxniddd * zeff+dxnidd * dzeffdd+dxnidd * dzeffdd+xni * dzeffddd;
    Real dxneddt = dxniddt * zeff+dxnidd * dzeffdt+dxnidt * dzeffdd+xni * dzeffddt;
    Real dxnedda = dxnidda * zeff+dxnidd * dzeffda+dxnida * dzeffdd+xni * dzeffdda;
    Real dxneddz = dxniddz * zeff+dxnidd * dzeffdz+dxnidz * dzeffdd+xni * dzeffddz;
    Real dxnedtt = dxnidtt * zeff+dxnidt * dzeffdt+dxnidt * dzeffdt+xni * dzeffdtt;
    Real dxnedta = dxnidta * zeff+dxnidt * dzeffda+dxnida * dzeffdt+xni * dzeffdta;
    Real dxnedtz = dxnidtz * zeff+dxnidt * dzeffdz+dxnidz * dzeffdt+xni * dzeffdtz;
    Real dxnedaa = dxnidaa * zeff+dxnida * dzeffda+dxnida * dzeffda+xni * dzeffdaa;
    Real dxnedaz = dxnidaz * zeff+dxnida * dzeffdz+dxnidz * dzeffda+xni * dzeffdaz;
    Real dxnedzz = dxnidzz * zeff+dxnidz * dzeffdz+dxnidz * dzeffdz+xni * dzeffdzz;

    // first derivatives of the fermi integral electron number densities
    Real dxneferdd = dxnefer_deta * detadd;
    Real dxneferdt = dxnefer_deta * detadt + dxnefer_dbeta * dbetadt;
    Real dxneferda = dxnefer_deta * detada;
    Real dxneferdz = dxnefer_deta * detadz;

    // second derivatives
    Real dxneferddd = dxnefer_deta2 * detadd * detadd + dxnefer_deta * detaddd;
    Real dxneferddt = dxnefer_deta2 * detadt * detadd +
                      dxnefer_deta_dbeta * dbetadt * detadd +
                      dxnefer_deta * detaddt;
    Real dxneferdda = dxnefer_deta2 * detada * detadd + dxnefer_deta * detadda;
    Real dxneferddz = dxnefer_deta2 * detadz * detadd + dxnefer_deta * detaddz;
    Real dxneferdtt = dxnefer_deta2 * detadt * detadt + dxnefer_deta * detadtt +
                      2.0_rt * dxnefer_deta_dbeta * detadt * dbetadt +
                      dxnefer_dbeta2 * dbetadt * dbetadt;
    Real dxneferdta = dxnefer_deta2 * detada * detadt +
                      dxnefer_deta_dbeta * dbetadt * detada +
                      dxnefer_deta * detadta;
    Real dxneferdtz = dxnefer_deta2 * detadz * detadt +
                      dxnefer_deta_dbeta * dbetadt * detadz +
                      dxnefer_deta * detadtz;
    Real dxneferdaa = dxnefer_deta2 * detada * detada + dxnefer_deta * detadaa;
    Real dxneferdaz = dxnefer_deta2 * detadz * detada + dxnefer_deta * detadaz;
    Real dxneferdzz = dxnefer_deta2 * detadz * detadz + dxnefer_deta * detadzz;

    // first derivatives of the fermi integral positron number densities
    Real dxnpferdd = dxnpfer_deta * detadd;
    Real dxnpferdt = dxnpfer_deta * detadt + dxnpfer_dbeta * dbetadt;
    Real dxnpferda = dxnpfer_deta * detada;
    Real dxnpferdz = dxnpfer_deta * detadz;

    // second derivatives
    Real dxnpferddd = dxnpfer_deta2 * detadd * detadd + dxnpfer_deta * detaddd;
    Real dxnpferddt = dxnpfer_deta2 * detadt * detadd +
                      dxnpfer_deta_dbeta * dbetadt * detadd +
                      dxnpfer_deta * detaddt;
    Real dxnpferdda = dxnpfer_deta2 * detada * detadd + dxnpfer_deta * detadda;
    Real dxnpferddz = dxnpfer_deta2 * detadz * detadd + dxnpfer_deta * detaddz;
    Real dxnpferdtt = dxnpfer_deta2 * detadt * detadt + dxnpfer_deta * detadtt +
                      2.0_rt * dxnpfer_deta_dbeta * detadt * dbetadt +
                      dxnpfer_dbeta2 * dbetadt * dbetadt;
    Real dxnpferdta = dxnpfer_deta2 * detada * detadt +
                      dxnpfer_deta_dbeta * dbetadt * detada +
                      dxnpfer_deta * detadta;
    Real dxnpferdtz = dxnpfer_deta2 * detadz * detadt +
                      dxnpfer_deta_dbeta * dbetadt * detadz +
                      dxnpfer_deta * detadtz;
    Real dxnpferdaa = dxnpfer_deta2 * detada * detada + dxnpfer_deta * detadaa;
    Real dxnpferdaz = dxnpfer_deta2 * detadz * detada + dxnpfer_deta * detadaz;
    Real dxnpferdzz = dxnpfer_deta2 * detadz * detadz + dxnpfer_deta * detadzz;

    // now get the pressure and energy
    // for the electrons

    dfermi(1.5_rt, etaele, beta, f32, f32eta, f32beta, f32eta2, f32beta2, f32etabeta);
    Real f52, f52eta, f52beta, f52eta2, f52beta2, f52etabeta;
    dfermi(2.5_rt, etaele, beta, f52, f52eta, f52beta, f52eta2, f52beta2, f52etabeta);

    // pressure in erg/cm**3
    yy   = pconst * beta52;
    ww   = pconst * 2.5_rt * beta32;
    dum1 = f32 + 0.5_rt * beta * f52;
    dum2 = f32beta + 0.5_rt * f52 + 0.5_rt * beta * f52beta;
    Real dum3 = f32eta + 0.5_rt * beta * f52eta;

    Real pele         = yy * dum1;
    Real dpele_deta   = yy * dum3;
    Real dpele_dbeta  = ww * dum1 + yy * dum2;
    Real dpele_deta2  = yy * (f32eta2 + 0.5_rt * beta * f52eta2);
    Real dpele_dbeta2 = pconst * beta12 * 3.75_rt * dum1 + 2.0_rt * ww * dum2 +
                        yy * (f32beta2 + f52beta + 0.5_rt * beta * f52beta2);
    Real dpele_deta_dbeta = ww * dum3 + yy * (f32etabeta +0.5_rt * f52eta +
                                              0.5_rt * beta * f52etabeta);

    // first derivatives of the electron pressure
    Real dpeledd = dpele_deta * detadd;
    Real dpeledt = dpele_deta * detadt + dpele_dbeta * dbetadt;
    Real dpeleda = dpele_deta * detada;
    Real dpeledz = dpele_deta * detadz;

    // second derivatives
    Real dpeleddd = dpele_deta2 * detadd * detadd + dpele_deta * detaddd;
    Real dpeleddt = dpele_deta2 * detadt * detadd +
                    dpele_deta_dbeta * dbetadt * detadd +
                    dpele_deta * detaddt;
    Real dpeledda = dpele_deta2 * detada * detadd + dpele_deta * detadda;
    Real dpeleddz = dpele_deta2 * detadz * detadd + dpele_deta * detaddz;
    Real dpeledtt = dpele_deta2 * detadt * detadt + dpele_deta * detadtt +
                    2.0_rt * dpele_deta_dbeta * detadt * dbetadt +
                    dpele_dbeta2 * dbetadt * dbetadt;
    Real dpeledta = dpele_deta2 * detada * detadt +
                    dpele_deta_dbeta * dbetadt * detada +
                    dpele_deta * detadta;
    Real dpeledtz = dpele_deta2 * detadz * detadt +
                    dpele_deta_dbeta * dbetadt * detadz +
                    dpele_deta * detadtz;
    Real dpeledaa = dpele_deta2 * detada * detada + dpele_deta * detadaa;
    Real dpeledaz = dpele_deta2 * detadz * detada + dpele_deta * detadaz;
    Real dpeledzz = dpele_deta2 * detadz * detadz + dpele_deta * detadzz;

    // energy in erg/cm**3
    yy   = econst * beta52;
    ww   = econst * 2.5_rt * beta32;
    dum1 = f32 + beta * f52;
    dum2 = f32beta + f52 + beta * f52beta;
    dum3 = f32eta + beta * f52eta;

    Real eele        = yy * dum1;
    Real deele_deta  = yy * dum3;
    Real deele_dbeta = ww * dum1 + yy * dum2;
    Real deele_deta2  = yy * (f32eta2 + beta * f52eta2);
    Real deele_dbeta2 = econst * beta12 * 3.75_rt * dum1 + 2.0_rt * ww * dum2 +
                        yy * (f32beta2 + 2.0_rt * f52beta + beta * f52beta2);
    Real deele_deta_dbeta = ww * dum3 +
                            yy * (f32etabeta + f52eta + beta * f52etabeta);

    // first derivatives of the electron energy
    Real deeledd = deele_deta * detadd;
    Real deeledt = deele_deta * detadt + deele_dbeta * dbetadt;
    Real deeleda = deele_deta * detada;
    Real deeledz = deele_deta * detadz;

    // second derivatives
    Real deeleddd = deele_deta2 * detadd * detadd + deele_deta * detaddd;
    Real deeleddt = deele_deta2 * detadt * detadd +
                    deele_deta_dbeta * dbetadt * detadd +
                    deele_deta * detaddt;
    Real deeledda = deele_deta2 * detada * detadd + deele_deta * detadda;
    Real deeleddz = deele_deta2 * detadz * detadd + deele_deta * detaddz;
    Real deeledtt = deele_deta2 * detadt * detadt + deele_deta * detadtt +
                    2.0_rt * deele_deta_dbeta * detadt * dbetadt +
                    deele_dbeta2 * dbetadt * dbetadt;
    Real deeledta = deele_deta2 * detada * detadt +
                    deele_deta_dbeta * dbetadt * detada +
                    deele_deta * detadta;
    Real deeledtz = deele_deta2 * detadz * detadt +
                    deele_deta_dbeta * dbetadt * detadz +
                    deele_deta * detadtz;
    Real deeledaa = deele_deta2 * detada * detada + deele_deta * detadaa;
    Real deeledaz = deele_deta2 * detadz * detada + deele_deta * detadaz;
    Real deeledzz = deele_deta2 * detadz * detadz + deele_deta * detadzz;

    // for the positrons
    Real ppos              = 0.0_rt;
    Real dppos_detap       = 0.0_rt;
    Real dppos_dbeta       = 0.0_rt;
    Real dppos_detap2      = 0.0_rt;
    Real dppos_dbeta2      = 0.0_rt;
    Real dppos_detap_dbeta = 0.0_rt;
    Real epos              = 0.0_rt;
    Real depos_detap       = 0.0_rt;
    Real depos_dbeta       = 0.0_rt;
    Real depos_detap2      = 0.0_rt;
    Real depos_dbeta2      = 0.0_rt;
    Real depos_detap_dbeta = 0.0_rt;

    if (beta > positron_start) {
        dfermi(1.5_rt, etapos, beta, f32, f32eta, f32beta, f32eta2, f32beta2, f32etabeta);
        dfermi(2.5_rt, etapos, beta, f52, f52eta, f52beta, f52eta2, f52beta2, f52etabeta);

        // pressure
        yy   = pconst * beta52;
        ww   = pconst * 2.5_rt * beta32;
        dum1 = f32 + 0.5_rt * beta * f52;
        dum2 = f32beta + 0.5_rt * f52 + 0.5_rt * beta * f52beta;
        dum3 = f32eta + 0.5_rt * beta * f52eta;

        ppos         = yy * dum1;
        dppos_detap  = yy * dum3;
        dppos_dbeta  = ww * dum1 + yy * dum2;
        dppos_detap2 = yy * (f32eta2 + 0.5_rt * beta * f52eta2);
        dppos_dbeta2 = pconst * beta12 * 3.75_rt * dum1 + 2.0_rt * ww * dum2 +
                       yy * (f32beta2 + f52beta + 0.5_rt * beta * f52beta2);
        dppos_detap_dbeta = ww * dum3 + yy * (f32etabeta + 0.5_rt * f52eta +
                                              0.5_rt * beta * f52etabeta);

        // energy
        yy   = econst * beta52;
        ww   = econst * 2.5_rt * beta32;
        dum1 = f32 + beta * f52;
        dum2 = f32beta + f52 + beta * f52beta;
        dum3 = f32eta + beta * f52eta;

        epos        = yy * dum1;
        depos_detap  = yy * dum3;
        depos_dbeta = ww * dum1 + yy * dum2;
        depos_detap2  = yy * (f32eta2 + beta * f52eta2);
        depos_dbeta2 = econst * beta12 * 3.75_rt * dum1 + 2.0_rt * ww * dum2 +
                       yy * (f32beta2 + 2.0_rt * f52beta + beta * f52beta2);
        depos_detap_dbeta = ww * dum3 +
                            yy * (f32etabeta + f52eta + beta * f52etabeta);
    }

    // convert the etap derivatives to eta derivatives
    // all derived from the operator dxp = dxp/detap detap + dxp/dbeta dbeta

    Real dppos_deta  = dppos_detap * detap_deta;
    dppos_dbeta = dppos_dbeta + dppos_detap * detap_dbeta;
    Real dppos_deta2 = dppos_detap2 * detap_deta * detap_deta +
                       dppos_detap * detap_deta2;
    dppos_dbeta2 = dppos_dbeta2 +
                   2.0_rt * dppos_detap_dbeta * detap_dbeta +
                   dppos_detap2 * detap_dbeta * detap_dbeta +
                   dppos_detap * detap_dbeta2;
    Real dppos_deta_dbeta = dppos_detap2 * detap_dbeta * detap_deta +
                            dppos_detap_dbeta * detap_deta +
                            dppos_detap * detap_deta_dbeta;

    // first derivatives of the positron pressure
    Real dpposdd = dppos_deta * detadd;
    Real dpposdt = dppos_deta * detadt + dppos_dbeta * dbetadt;
    Real dpposda = dppos_deta * detada;
    Real dpposdz = dppos_deta * detadz;

    // second derivatives
    Real dpposddd = dppos_deta2 * detadd * detadd + dppos_deta * detaddd;
    Real dpposddt = dppos_deta2 * detadt * detadd +
                    dppos_deta_dbeta * dbetadt * detadd +
                    dppos_deta * detaddt;
    Real dpposdda = dppos_deta2 * detada * detadd + dppos_deta * detadda;
    Real dpposddz = dppos_deta2 * detadz * detadd + dppos_deta * detaddz;
    Real dpposdtt = dppos_deta2 * detadt * detadt + dppos_deta * detadtt +
                    2.0_rt * dppos_deta_dbeta * detadt * dbetadt +
                    dppos_dbeta2 * dbetadt * dbetadt;
    Real dpposdta = dppos_deta2 * detada * detadt +
                    dppos_deta_dbeta * dbetadt * detada +
                    dppos_deta * detadta;
    Real dpposdtz = dppos_deta2 * detadz * detadt +
                    dppos_deta_dbeta * dbetadt * detadz +
                    dppos_deta * detadtz;
    Real dpposdaa = dppos_deta2 * detada * detada + dppos_deta * detadaa;
    Real dpposdaz = dppos_deta2 * detadz * detada + dppos_deta * detadaz;
    Real dpposdzz = dppos_deta2 * detadz * detadz + dppos_deta * detadzz;

    // convert the etap derivatives to eta derivatives
    // all derived from the operator dxp = dxp/detap detap + dxp/dbeta dbeta

    Real depos_deta  = depos_detap * detap_deta;
    depos_dbeta = depos_dbeta + depos_detap * detap_dbeta;
    Real depos_deta2 = depos_detap2 * detap_deta * detap_deta +
                       depos_detap * detap_deta2;
    depos_dbeta2 = depos_dbeta2 +
                   2.0_rt * depos_detap_dbeta * detap_dbeta +
                   depos_detap2 * detap_dbeta * detap_dbeta +
                   depos_detap * detap_dbeta2;
    Real depos_deta_dbeta = depos_detap2 * detap_dbeta * detap_deta +
                            depos_detap_dbeta * detap_deta +
                            depos_detap * detap_deta_dbeta;

    // first derivatives of the positron energy
    Real deposdd = depos_deta * detadd;
    Real deposdt = depos_deta * detadt + depos_dbeta * dbetadt;
    Real deposda = depos_deta * detada;
    Real deposdz = depos_deta * detadz;

    // second derivatives
    Real deposddd = depos_deta2 * detadd * detadd + depos_deta * detaddd;
    Real deposddt = depos_deta2 * detadt * detadd +
                    depos_deta_dbeta * dbetadt * detadd +
                    depos_deta * detaddt;
    Real deposdda = depos_deta2 * detada * detadd + depos_deta * detadda;
    Real deposddz = depos_deta2 * detadz * detadd + depos_deta * detaddz;
    Real deposdtt = depos_deta2 * detadt * detadt + depos_deta * detadtt +
                    2.0_rt * depos_deta_dbeta * detadt * dbetadt +
                    depos_dbeta2 * dbetadt * dbetadt;
    Real deposdta = depos_deta2 * detada * detadt +
                    depos_deta_dbeta * dbetadt * detada +
                    depos_deta * detadta;
    Real deposdtz = depos_deta2 * detadz * detadt +
                    depos_deta_dbeta * dbetadt * detadz +
                    depos_deta * detadtz;
    Real deposdaa = depos_deta2 * detada * detada + depos_deta * detadaa;
    Real deposdaz = depos_deta2 * detadz * detada + depos_deta * detadaz;
    Real deposdzz = depos_deta2 * detadz * detadz + depos_deta * detadzz;

    // electron+positron pressure and its derivatives
    // note: at high temperatures and low densities, dpepdd is very small
    // and can go negative, so limit it to be positive definite

    Real pep     = pele + ppos;
    Real dpepdd  = std::max(dpeledd + dpposdd, 1.0e-30_rt);
    Real dpepdt  = dpeledt + dpposdt;
    Real dpepda  = dpeleda + dpposda;
    Real dpepdz  = dpeledz + dpposdz;
    Real dpepddd = dpeleddd + dpposddd;
    Real dpepddt = dpeleddt + dpposddt;
    Real dpepdda = dpeledda + dpposdda;
    Real dpepddz = dpeleddz + dpposddz;
    Real dpepdtt = dpeledtt + dpposdtt;
    Real dpepdta = dpeledta + dpposdta;
    Real dpepdtz = dpeledtz + dpposdtz;
    Real dpepdaa = dpeledaa + dpposdaa;
    Real dpepdaz = dpeledaz + dpposdaz;
    Real dpepdzz = dpeledzz + dpposdzz;

    // electron+positron thermal energy and its derivatives
    Real eep     = eele + epos;
    Real deepdd  = deeledd + deposdd;
    Real deepdt  = deeledt + deposdt;
    Real deepda  = deeleda + deposda;
    Real deepdz  = deeledz + deposdz;
    Real deepddd = deeleddd + deposddd;
    Real deepddt = deeleddt + deposddt;
    Real deepdda = deeledda + deposdda;
    Real deepddz = deeleddz + deposddz;
    Real deepdtt = deeledtt + deposdtt;
    Real deepdta = deeledta + deposdta;
    Real deepdtz = deeledtz + deposdtz;
    Real deepdaa = deeledaa + deposdaa;
    Real deepdaz = deeledaz + deposdaz;
    Real deepdzz = deeledzz + deposdzz;

    // electron entropy in erg/gr/kelvin and its derivatives
    y = kerg * deni;
    Real sele = ((pele + eele) * kti - etaele * xnefer) * y;

    // first derivatives
    Real dseledd = ((dpeledd + deeledd) * kti - detadd * xnefer -
                    etaele * dxneferdd) * y - sele * deni;
    Real dseledt = ((dpeledt + deeledt) * kti - (pele + eele) / (kt * state.T) -
                    detadt * xnefer - etaele * dxneferdt) * y;
    Real dseleda = ((dpeleda + deeleda) * kti - detada * xnefer -
                    etaele * dxneferda) * y;
    Real dseledz = ((dpeledz + deeledz) * kti - detadz * xnefer -
                    etaele * dxneferdz) * y;

    // second derivatives
    Real dseleddd = ((dpeleddd + deeleddd) * kti - detaddd * xnefer -
                     2.0_rt * detadd * dxneferdd - etaele * dxneferddd) * y -
                    2.0_rt * ((dpeledd + deeledd) * kti - detadd * xnefer -
                              etaele * dxneferdd) * y * deni +
                    2.0_rt * sele * deni * deni;
    Real dseleddt = ((dpeleddt + deeleddt) * kti -
                     (dpeledd + deeledd) * kti / state.T -
                     detaddt * xnefer - detadd * dxneferdt -
                     detadt * dxneferdd - etaele * dxneferddt) * y -
                    dseledt * deni;
    Real dseledda = ((dpeledda + deeledda) * kti -
                     detadda * xnefer - detadd * dxneferda -
                     detada * dxneferdd - etaele * dxneferdda) * y -
                    dseleda * deni;
    Real dseleddz = ((dpeleddz + deeleddz) * kti -
                     detaddz * xnefer - detadd * dxneferdz -
                     detadz * dxneferdd - etaele * dxneferddz) * y -
                    dseledz * deni;
    Real dseledtt = ((dpeledtt + deeledtt) * kti -
                     2.0_rt * (dpeledt + deeledt) * kti / state.T +
                     2.0_rt * (pele + eele) * kti / (state.T * state.T) -
                     detadtt * xnefer - 2.0_rt * detadt * dxneferdt -
                     etaele * dxneferdtt) * y;
    Real dseledta = ((dpeledta + deeledta) * kti -
                     (dpeleda + deeleda) * kti / state.T -
                     detadta * xnefer - detadt * dxneferda -
                     detada * dxneferdt - etaele * dxneferdta) * y;
    Real dseledtz = ((dpeledtz + deeledtz) * kti -
                     (dpeledz + deeledz) * kti / state.T -
                     detadtz * xnefer - detadt * dxneferdz -
                     detadz * dxneferdt - etaele * dxneferdtz) * y;
    Real dseledaa = ((dpeledaa + deeledaa) * kti - detadaa * xnefer -
                     2.0_rt * detada * dxneferda - etaele * dxneferdaa) * y;
    Real dseledaz = ((dpeledaz + deeledaz) * kti - detadaz * xnefer -
                     detada * dxneferdz -
                     detadz * dxneferda - etaele * dxneferdaz) * y;
    Real dseledzz = ((dpeledzz + deeledzz) * kti - detadzz * xnefer -
                     2.0_rt * detadz * dxneferdz - etaele * dxneferdzz) * y;

    // positron entropy in erg/gr/kelvin and its derivatives
    Real spos = ((ppos + epos) / kt - etapos * xnpfer) * y;

    // first derivatives
    Real dsposdd = ((dpposdd + deposdd) * kti -
                    detap_deta * detadd * xnpfer -
                    etapos * dxnpferdd) * y - spos * deni;
    Real dsposdt = ((dpposdt + deposdt) * kti - (ppos + epos) / (kt * state.T) -
                    (detap_deta * detadt + detap_dbeta * dbetadt) * xnpfer -
                    etapos * dxnpferdt) * y;
    Real dsposda = ((dpposda + deposda) * kti -
                    detap_deta * detada * xnpfer -
                    etapos * dxnpferda) * y;
    Real dsposdz = ((dpposdz + deposdz) * kti - detap_deta * detadz * xnpfer -
                    etapos * dxnpferdz) * y;

    // second derivatives
    Real dsposddd = ((dpposddd + deposddd) * kti -
                     detap_deta * detaddd * xnpfer -
                     2.0_rt * detap_deta * detadd * dxnpferdd -
                     etapos * dxnpferddd) * y -
                    2.0_rt * ((dpposdd + deposdd) * kti -
                              detap_deta * detadd * xnpfer -
                              etapos * dxnpferdd) * y * deni +
                    2.0_rt * spos * deni * deni;
    Real dsposddt = ((dpposddt + deposddt) * kti -
                     (dpposdd + deposdd) * kti / state.T -
                     detap_deta * detaddt * xnpfer -
                     detap_deta * detadd * dxnpferdt -
                     (detap_deta * detadt + detap_dbeta * dbetadt) * dxnpferdd -
                     etapos * dxnpferddt) * y -
                    dsposdt * deni;
    Real dsposdda = ((dpposdda + deposdda) * kti -
                     detap_deta * detadda * xnpfer -
                     detap_deta * detadd * dxnpferda -
                     detap_deta * detada * dxnpferdd -
                     etapos * dxnpferdda) * y -
                    dsposda * deni;
    Real dsposddz = ((dpposddz + deposddz) * kti -
                     detap_deta * detaddz * xnpfer -
                     detap_deta * detadd * dxnpferdz -
                     detap_deta * detadz * dxnpferdd -
                     etapos * dxnpferddz) * y -
                    dsposdz * deni;

    Real dsposdtt = ((dpposdtt + deposdtt) * kti -
                     2.0_rt * (dpposdt + deposdt) * kti / state.T +
                     2.0_rt * (ppos + epos) * kti / (state.T * state.T) -
                     (detap_deta2 * detadt * detadt +
                      2.0_rt * detap_deta_dbeta * detadt * dbetadt +
                      detap_deta * detadtt +
                      detap_dbeta2 * dbetadt * dbetadt) * xnpfer -
                     2.0_rt * (detap_deta * detadt +
                               detap_dbeta * dbetadt) * dxnpferdt -
                     etapos * dxnpferdtt) * y;

    Real dsposdta = ((dpposdta + deposdta) * kti -
                     (dpposda + deposda) * kti / state.T -
                     (detap_deta2 * detadt * detada +
                      detap_deta_dbeta * dbetadt * detada +
                      detap_deta * detadta) * xnpfer -
                     (detap_deta * detadt +
                      detap_dbeta * dbetadt) * dxnpferda -
                     detap_deta * detada * dxnpferdt -
                     etapos * dxnpferdta) * y;
    Real dsposdtz = ((dpposdtz + deposdtz) * kti -
                     (dpposdz + deposdz) * kti / state.T -
                     (detap_deta2 * detadt * detadz +
                      detap_deta_dbeta * dbetadt * detadz +
                      detap_deta * detadtz) * xnpfer -
                     (detap_deta * detadt +
                      detap_dbeta * dbetadt) * dxnpferdz -
                     detap_deta * detadz * dxnpferdt -
                     etapos * dxnpferdtz) * y;
    Real dsposdaa = ((dpposdaa + deposdaa) * kti -
                     detap_deta * detadaa * xnpfer -
                     2.0_rt * detap_deta * detada * dxnpferda -
                     etapos * dxnpferdaa) * y;
    Real dsposdaz = ((dpposdaz + deposdaz) * kti -
                     detap_deta * detadaz * xnpfer -
                     detap_deta * detada * dxnpferdz -
                     detap_deta * detadz * dxnpferda -
                     etapos * dxnpferdaz) * y;
    Real dsposdzz = ((dpposdzz + deposdzz) * kti -
                     detap_deta * detadzz * xnpfer -
                     2.0_rt * detap_deta * detadz * dxnpferdz -
                     etapos * dxnpferdzz) * y;

    // and their sum
    Real sep      = sele + spos;
    Real dsepdd   = dseledd + dsposdd;
    Real dsepdt   = dseledt + dsposdt;
    Real dsepda   = dseleda + dsposda;
    Real dsepdz   = dseledz + dsposdz;
    Real dsepddd  = dseleddd + dsposddd;
    Real dsepddt  = dseleddt + dsposddt;
    Real dsepdda  = dseledda + dsposdda;
    Real dsepddz  = dseleddz + dsposddz;
    Real dsepdtt  = dseledtt + dsposdtt;
    Real dsepdta  = dseledta + dsposdta;
    Real dsepdtz  = dseledtz + dsposdtz;
    Real dsepdaa  = dseledaa + dsposdaa;
    Real dsepdaz  = dseledaz + dsposdaz;
    Real dsepdzz  = dseledzz + dsposdzz;

    // adjust for the rest mass energy of the positrons
    y        = 2.0_rt * mecc;
    epos     = epos     + y * xnpfer;
    deposdd  = deposdd  + y * dxnpferdd;
    deposdt  = deposdt  + y * dxnpferdt;
    deposda  = deposda  + y * dxnpferda;
    deposdz  = deposdz  + y * dxnpferdz;
    deposddd = deposddd + y * dxnpferddd;
    deposddt = deposddt + y * dxnpferddt;
    deposdda = deposdda + y * dxnpferdda;
    deposddz = deposddz + y * dxnpferddz;
    deposdtt = deposdtt + y * dxnpferdtt;
    deposdta = deposdta + y * dxnpferdta;
    deposdtz = deposdtz + y * dxnpferdtz;
    deposdaa = deposdaa + y * dxnpferdaa;
    deposdaz = deposdaz + y * dxnpferdaz;
    deposdzz = deposdzz + y * dxnpferdzz;

    // and resum
    deepdd  = deeledd + deposdd;
    deepdt  = deeledt + deposdt;
    deepda  = deeleda + deposda;
    deepdz  = deeledz + deposdz;
    deepddd = deeleddd + deposddd;
    deepddt = deeleddt + deposddt;
    deepdda = deeledda + deposdda;
    deepddz = deeleddz + deposddz;
    deepdtt = deeledtt + deposdtt;
    deepdta = deeledta + deposdta;
    deepdtz = deeledtz + deposdtz;
    deepdaa = deeledaa + deposdaa;
    deepdaz = deeledaz + deposdaz;
    deepdzz = deeledzz + deposdzz;

    // convert the electron-positron thermal energy in erg/cm**3
    // to a specific thermal energy in erg/gr

    eele     = eele * deni;
    deeledd  = (deeledd - eele) * deni;
    deeledt  = deeledt * deni;
    deeleda  = deeleda * deni;
    deeledz  = deeledz * deni;
    deeleddd = (deeleddd - 2.0_rt * deeledd) * deni;
    deeleddt = (deeleddt - deeledt) * deni;
    deeledda = (deeledda - deeleda) * deni;
    deeleddz = (deeleddz - deeledz) * deni;
    deeledtt = deeledtt * deni;
    deeledta = deeledta * deni;
    deeledtz = deeledtz * deni;
    deeledaa = deeledaa * deni;
    deeledaz = deeledaz * deni;
    deeledzz = deeledzz * deni;

    epos     = epos * deni;
    deposdd  = (deposdd - epos) * deni;
    deposdt  = deposdt * deni;
    deposda  = deposda * deni;
    deposdz  = deposdz * deni;
    deposddd = (deposddd - 2.0_rt * deposdd) * deni;
    deposddt = (deposddt - deposdt) * deni;
    deposdda = (deposdda - deposda) * deni;
    deposddz = (deposddz - deposdz) * deni;
    deposdtt = deposdtt * deni;
    deposdta = deposdta * deni;
    deposdtz = deposdtz * deni;
    deposdaa = deposdaa * deni;
    deposdaz = deposdaz * deni;
    deposdzz = deposdzz * deni;

    // and resum
    deepdd  = deeledd + deposdd;
    deepdt  = deeledt + deposdt;
    deepda  = deeleda + deposda;
    deepdz  = deeledz + deposdz;
    deepddd = deeleddd + deposddd;
    deepddt = deeleddt + deposddt;
    deepdda = deeledda + deposdda;
    deepddz = deeleddz + deposddz;
    deepdtt = deeledtt + deposdtt;
    deepdta = deeledta + deposdta;
    deepdtz = deeledtz + deposdtz;
    deepdaa = deeledaa + deposdaa;
    deepdaz = deeledaz + deposdaz;
    deepdzz = deeledzz + deposdzz;

    // Fill in the state with the final values
    state.p = pele + ppos;
    state.e = eele + epos;
    state.s = sele + spos;

    state.dpdT = dpposdt + dpeledt;
    state.dpdr = dpposdd + dpeledd;

    state.dsdT = dsposdt + dseledt;
    state.dsdr = dsposdd + dseledd;

    state.dedT = deposdt + deeledt;
    state.dedr = deposdd + deeledd;

    state.dpdrdT = dpposddt + dpeleddt;
    state.dsdrdT = dsposddt + dseleddt;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
eos_timmes_t timmes_eos (Real rho, Real T)
{
    eos_timmes_t state;

    state.abar = 1.0_rt;
    state.zbar = 1.0_rt;
    state.T = T;
    state.rho = rho;

    Real ytot1   = 1.0e0_rt / state.abar;
    Real xni     = helmholtz::avo_eos * ytot1 * rho;
    Real eta_guess = etages(xni, state.zbar, T);

    bool eosfail = false;

    Real fk, dfk;
    for (int n = 0; n < 100; ++n) {
        xneroot(0, state, eta_guess, fk, dfk);

        if (dfk == 0.0_rt) {
            amrex::Error("Helmholtz EOS table convergence failure");
        }
        Real ratio = fk / dfk;
        Real eta_guess_new = eta_guess - ratio;
        Real z = std::abs((eta_guess_new - eta_guess) / eta_guess);
        eta_guess = eta_guess_new;
        const Real eostol = 1.0e-13_rt;
        const Real fpmin = 1.0e-14_rt;
        if (z < eostol || std::abs(ratio) <= fpmin) {
            break;
        }
    }

    // with the converged values, get the energy, pressure, and entropy
    xneroot(1, state, eta_guess, fk, dfk);

    return state;
}

#endif
