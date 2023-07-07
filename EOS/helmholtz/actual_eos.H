#ifndef ACTUAL_EOS_H
#define ACTUAL_EOS_H

#include <string>
#include <iostream>
#include <sstream>
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_ParallelDescriptor.H>
#include <extern_parameters.H>
#include <fundamental_constants.H>
#include <eos_type.H>
#include <eos_data.H>
#include <actual_eos_data.H>
#include <cmath>
#include <vector>

using namespace eos_rp;

// Frank Timmes Helmholtz based Equation of State
// https://cococubed.com/

// given a temperature temp [K], density den [g/cm**3], and a composition
// characterized by abar and zbar, this routine returns most of the other
// thermodynamic quantities. of prime interest is the pressure [erg/cm**3],
// specific thermal energy [erg/gr], the entropy [erg/g/K], along with
// their derivatives with respect to temperature, density, abar, and zbar.
// other quantites such the normalized chemical potential eta (plus its
// derivatives), number density of electrons and positron pair (along
// with their derivatives), adiabatic indices, specific heats, and
// relativistically correct sound speed are also returned.
// 
// this routine assumes planckian photons, an ideal gas of ions,
// and an electron-positron gas with an arbitrary degree of relativity
// and degeneracy. interpolation in a table of the helmholtz free energy
// is used to return the electron-positron thermodynamic quantities.
// all other derivatives are analytic.
// 
// references: cox & giuli chapter 24 ; timmes & swesty apj 1999

const std::string eos_name = "helmholtz";

struct eos_timmes_t : eos_extra_t {
    Real dpdrdT{};
    Real dsdrdT{};
};

// quintic hermite polynomial functions
// psi0 and its derivatives
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real psi0 (Real z)
{
    return z * z * z * (z * (-6.0e0_rt * z + 15.0e0_rt) -10.0e0_rt) + 1.0e0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real dpsi0 (Real z)
{
    return z * z * (z * (-30.0e0_rt*z + 60.0e0_rt) - 30.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real ddpsi0 (Real z)
{
    return z * (z * (-120.0e0_rt * z + 180.0e0_rt) - 60.0e0_rt);
}

// psi1 and its derivatives
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real psi1 (Real z)
{
    return z * ( z * z * (z * (-3.0e0_rt * z + 8.0e0_rt) - 6.0e0_rt) + 1.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real dpsi1 (Real z)
{
    return z * z * ( z * (-15.0e0_rt * z + 32.0e0_rt) - 18.0e0_rt) + 1.0e0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real ddpsi1 (Real z)
{
    return z * (z * (-60.0e0_rt * z + 96.0e0_rt) - 36.0e0_rt);
}

// psi2 and its derivatives
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real psi2 (Real z)
{
    return 0.5e0_rt * z * z * ( z * ( z * (-z + 3.0e0_rt) - 3.0e0_rt) + 1.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real dpsi2 (Real z)
{
    return 0.5e0_rt * z * (z * (z * (-5.0e0_rt * z + 12.0e0_rt) - 9.0e0_rt) + 2.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real ddpsi2 (Real z)
{
    return 0.5e0_rt * (z * ( z * (-20.0e0_rt * z + 36.0e0_rt) - 18.0e0_rt) + 2.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void fwt (const Real* fi, const Real* wt, Real* fwtr)
{
    fwtr[0] = fi[ 0]*wt[0] + fi[ 1]*wt[1] + fi[ 2]*wt[2] + fi[18]*wt[3] + fi[19]*wt[4] + fi[20]*wt[5];
    fwtr[1] = fi[ 3]*wt[0] + fi[ 5]*wt[1] + fi[ 7]*wt[2] + fi[21]*wt[3] + fi[23]*wt[4] + fi[25]*wt[5];
    fwtr[2] = fi[ 4]*wt[0] + fi[ 6]*wt[1] + fi[ 8]*wt[2] + fi[22]*wt[3] + fi[24]*wt[4] + fi[26]*wt[5];
    fwtr[3] = fi[ 9]*wt[0] + fi[10]*wt[1] + fi[11]*wt[2] + fi[27]*wt[3] + fi[28]*wt[4] + fi[29]*wt[5];
    fwtr[4] = fi[12]*wt[0] + fi[14]*wt[1] + fi[16]*wt[2] + fi[30]*wt[3] + fi[32]*wt[4] + fi[34]*wt[5];
    fwtr[5] = fi[13]*wt[0] + fi[15]*wt[1] + fi[17]*wt[2] + fi[31]*wt[3] + fi[33]*wt[4] + fi[35]*wt[5];
}

// cubic hermite polynomial functions
// psi0 & derivatives
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real xpsi0(Real z)
{
    return z * z * (2.0e0_rt * z - 3.0e0_rt) + 1.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real xdpsi0(Real z)
{
    return z * (6.0e0_rt * z - 6.0e0_rt);
}

// psi1 & derivatives
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real xpsi1(Real z)
{
    return z * (z * (z - 2.0e0_rt) + 1.0e0_rt);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real xdpsi1(Real z)
{
    return z * (3.0e0_rt * z - 4.0e0_rt) + 1.0e0_rt;
}

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

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void apply_electrons (T& state)
{
    using namespace helmholtz;

    // assume complete ionization
    [[maybe_unused]] Real ytot1 = 1.0e0_rt / state.abar;

    // define mu -- the total mean molecular weight including both electrons and ions
    state.mu = 1.0e0_rt / (1.0e0_rt / state.abar + 1.0e0_rt / state.mu_e);

    // enter the table with ye*den
    Real din = state.y_e * state.rho;

    // hash locate this temperature and density
    int jat = int((std::log10(state.T) - tlo) * tstpi) + 1;
    jat = amrex::max(1, amrex::min(jat, jtmax-1)) - 1;
    int iat = int((std::log10(din) - dlo) * dstpi) + 1;
    iat = amrex::max(1, amrex::min(iat, itmax-1)) - 1;

    Real fi[36];

    // access the table locations only once
    for (int i = 0; i < 9; ++i) {
        fi[i     ] = f[jat  ][iat  ][i]; // f, ft, ftt, fd, fdd, fdt, fddt, fdtt, fddtt
        fi[i +  9] = f[jat  ][iat+1][i];
        fi[i + 18] = f[jat+1][iat  ][i];
        fi[i + 27] = f[jat+1][iat+1][i];
    }

    // various differences
    Real xt  = amrex::max((state.T - t[jat]) * dti_sav[jat], 0.0e0_rt);
    Real xd  = amrex::max((din - d[iat]) * ddi_sav[iat], 0.0e0_rt);
    Real mxt = 1.0e0_rt - xt;
    Real mxd = 1.0e0_rt - xd;

    // the six density and six temperature basis functions
    Real sit[6];

    sit[0] = psi0(xt);
    sit[1] = psi1(xt) * dt_sav[jat];
    sit[2] = psi2(xt) * dt2_sav[jat];

    sit[3] =  psi0(mxt);
    sit[4] = -psi1(mxt) * dt_sav[jat];
    sit[5] =  psi2(mxt) * dt2_sav[jat];

    Real sid[6];

    sid[0] =  psi0(xd);
    sid[1] =  psi1(xd) * dd_sav[iat];
    sid[2] =  psi2(xd) * dd2_sav[iat];

    sid[3] =  psi0(mxd);
    sid[4] = -psi1(mxd) * dd_sav[iat];
    sid[5] =  psi2(mxd) * dd2_sav[iat];

    // derivatives of the weight functions
    Real dsit[6];

    dsit[0] =  dpsi0(xt) * dti_sav[jat];
    dsit[1] =  dpsi1(xt);
    dsit[2] =  dpsi2(xt) * dt_sav[jat];

    dsit[3] = -dpsi0(mxt) * dti_sav[jat];
    dsit[4] =  dpsi1(mxt);
    dsit[5] = -dpsi2(mxt) * dt_sav[jat];

    Real dsid[6];

    dsid[0] =  dpsi0(xd) * ddi_sav[iat];
    dsid[1] =  dpsi1(xd);
    dsid[2] =  dpsi2(xd) * dd_sav[iat];

    dsid[3] = -dpsi0(mxd) * ddi_sav[iat];
    dsid[4] =  dpsi1(mxd);
    dsid[5] = -dpsi2(mxd) * dd_sav[iat];

    // second derivatives of the weight functions
    Real ddsit[6];

    ddsit[0] =  ddpsi0(xt) * dt2i_sav[jat];
    ddsit[1] =  ddpsi1(xt) * dti_sav[jat];
    ddsit[2] =  ddpsi2(xt);

    ddsit[3] =  ddpsi0(mxt) * dt2i_sav[jat];
    ddsit[4] = -ddpsi1(mxt) * dti_sav[jat];
    ddsit[5] =  ddpsi2(mxt);

    // This array saves some subexpressions that go into
    // computing the biquintic polynomial. Instead of explicitly
    // constructing it in full, we'll use these subexpressions
    // and then compute the result as
    // (table data * temperature terms) * density terms.

    Real fwtr[6];

    fwt(fi, sit, fwtr);

    Real free = 0.e0_rt;
    Real df_d = 0.e0_rt;

    for (int i = 0; i <= 5; ++i) {
        // the free energy
        free = free + fwtr[i] * sid[i];

        // derivative with respect to density
        df_d = df_d + fwtr[i] * dsid[i];
    }

    fwt(fi, dsit, fwtr);

    Real df_t = 0.e0_rt;
    Real df_dt = 0.e0_rt;

    for (int i = 0; i <= 5; ++i) {
        // derivative with respect to temperature
        df_t += fwtr[i] * sid[i];

        // derivative with respect to temperature and density
        df_dt += fwtr[i] * dsid[i];
    }

    fwt(fi, ddsit, fwtr);

    Real df_tt = 0.e0_rt;
    for (int i = 0; i <= 5; ++i) {
        // derivative with respect to temperature**2
        df_tt = df_tt + fwtr[i] * sid[i];
    }

    // now get the pressure derivative with density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    sit[0] = xpsi0(xt);
    sit[1] = xpsi1(xt) * dt_sav[jat];

    sit[2] = xpsi0(mxt);
    sit[3] = -xpsi1(mxt) * dt_sav[jat];

    sid[0] = xpsi0(xd);
    sid[1] = xpsi1(xd) * dd_sav[iat];

    sid[2] = xpsi0(mxd);
    sid[3] = -xpsi1(mxd) * dd_sav[iat];

    // derivatives of weight functions
    dsit[0] = xdpsi0(xt) * dti_sav[jat];
    dsit[1] = xdpsi1(xt);

    dsit[2] = -xdpsi0(mxt) * dti_sav[jat];
    dsit[3] = xdpsi1(mxt);

    dsid[0] = xdpsi0(xd) * ddi_sav[iat];
    dsid[1] = xdpsi1(xd);

    dsid[2] = -xdpsi0(mxd) * ddi_sav[iat];
    dsid[3] = xdpsi1(mxd);

    // Reuse subexpressions that would go into computing the
    // cubic interpolation.
    Real wdt[16];

    for (int i = 0; i <= 3; ++i) {
        wdt[i     ] = sid[0] * sit[i];
        wdt[i +  4] = sid[1] * sit[i];
        wdt[i +  8] = sid[2] * sit[i];
        wdt[i + 12] = sid[3] * sit[i];
    }

    // Read in the tabular data for the pressure derivatives.
    // We have some freedom in how we store it in the local
    // array. We choose here to index it such that we can
    // immediately evaluate the cubic interpolant below as
    // fi * wdt, which ensures that we have the right combination
    // of grid points and derivatives at grid points to evaluate
    // the interpolation correctly. Alternate indexing schemes are
    // possible if we were to reorder wdt.
    fi[ 0] = dpdf[jat  ][iat  ][0];
    fi[ 1] = dpdf[jat  ][iat  ][1];
    fi[ 4] = dpdf[jat  ][iat  ][2];
    fi[ 5] = dpdf[jat  ][iat  ][3];

    fi[ 8] = dpdf[jat  ][iat+1][0];
    fi[ 9] = dpdf[jat  ][iat+1][1];
    fi[12] = dpdf[jat  ][iat+1][2];
    fi[13] = dpdf[jat  ][iat+1][3];

    fi[ 2] = dpdf[jat+1][iat  ][0];
    fi[ 3] = dpdf[jat+1][iat  ][1];
    fi[ 6] = dpdf[jat+1][iat  ][2];
    fi[ 7] = dpdf[jat+1][iat  ][3];

    fi[10] = dpdf[jat+1][iat+1][0];
    fi[11] = dpdf[jat+1][iat+1][1];
    fi[14] = dpdf[jat+1][iat+1][2];
    fi[15] = dpdf[jat+1][iat+1][3];

    // pressure derivative with density
    Real dpepdd = 0.0e0_rt;
    for (int i = 0; i <= 15; ++i) {
        dpepdd = dpepdd + fi[i] * wdt[i];
    }
    dpepdd = amrex::max(state.y_e * dpepdd, 0.0e0_rt);

    // Read in the tabular data for the electron chemical potential.
    fi[ 0] = ef[jat  ][iat  ][0];
    fi[ 1] = ef[jat  ][iat  ][1];
    fi[ 4] = ef[jat  ][iat  ][2];
    fi[ 5] = ef[jat  ][iat  ][3];

    fi[ 8] = ef[jat  ][iat+1][0];
    fi[ 9] = ef[jat  ][iat+1][1];
    fi[12] = ef[jat  ][iat+1][2];
    fi[13] = ef[jat  ][iat+1][3];

    fi[ 2] = ef[jat+1][iat  ][0];
    fi[ 3] = ef[jat+1][iat  ][1];
    fi[ 6] = ef[jat+1][iat  ][2];
    fi[ 7] = ef[jat+1][iat  ][3];

    fi[10] = ef[jat+1][iat+1][0];
    fi[11] = ef[jat+1][iat+1][1];
    fi[14] = ef[jat+1][iat+1][2];
    fi[15] = ef[jat+1][iat+1][3];

    // electron chemical potential etaele
    Real etaele = 0.0e0_rt;
    for (int i = 0; i <= 15; ++i) {
        etaele = etaele + fi[i] * wdt[i];
    }

    // Read in the tabular data for the number density.
    fi[ 0] = xf[jat  ][iat  ][0];
    fi[ 1] = xf[jat  ][iat  ][1];
    fi[ 4] = xf[jat  ][iat  ][2];
    fi[ 5] = xf[jat  ][iat  ][3];

    fi[ 8] = xf[jat  ][iat+1][0];
    fi[ 9] = xf[jat  ][iat+1][1];
    fi[12] = xf[jat  ][iat+1][2];
    fi[13] = xf[jat  ][iat+1][3];

    fi[ 2] = xf[jat+1][iat  ][0];
    fi[ 3] = xf[jat+1][iat  ][1];
    fi[ 6] = xf[jat+1][iat  ][2];
    fi[ 7] = xf[jat+1][iat  ][3];

    fi[10] = xf[jat+1][iat+1][0];
    fi[11] = xf[jat+1][iat+1][1];
    fi[14] = xf[jat+1][iat+1][2];
    fi[15] = xf[jat+1][iat+1][3];

    // electron + positron number densities
    Real xnefer = 0.0e0_rt;
    for (int i = 0; i <= 15; ++i) {
        xnefer = xnefer + fi[i] * wdt[i];
    }

    // the desired electron-positron thermodynamic quantities

    // dpepdd at high temperatures and low densities is below the
    // floating point limit of the subtraction of two large terms.
    // since state.dpdr doesn't enter the maxwell relations at all, use the
    // bicubic interpolation done above instead of this one
    Real x       = din * din;
    Real pele    = x * df_d;
    Real dpepdt  = x * df_dt;
    [[maybe_unused]] Real s       = dpepdd/state.y_e - 2.0e0_rt * din * df_d;
    [[maybe_unused]] Real dpepda  = -ytot1 * (2.0e0_rt * pele + s * din);
    [[maybe_unused]] Real dpepdz  = state.rho*ytot1*(2.0e0_rt * din * df_d  +  s);

    x            = state.y_e * state.y_e;
    Real sele    = -df_t * state.y_e;
    Real dsepdt  = -df_tt * state.y_e;
    Real dsepdd  = -df_dt * x;
    [[maybe_unused]] Real dsepda  = ytot1 * (state.y_e * df_dt * din - sele);
    [[maybe_unused]] Real dsepdz  = -ytot1 * (state.y_e * df_dt * state.rho  + df_t);

    Real eele    = state.y_e*free + state.T * sele;
    Real deepdt  = state.T * dsepdt;
    Real deepdd  = x * df_d + state.T * dsepdd;
    [[maybe_unused]] Real deepda  = -state.y_e * ytot1 * (free +  df_d * din) + state.T * dsepda;
    [[maybe_unused]] Real deepdz  = ytot1* (free + state.y_e * df_d * state.rho) + state.T * dsepdz;

    if constexpr (has_pressure<T>::value) {
        state.p    = state.p + pele;
        state.dpdT = state.dpdT + dpepdt;
        state.dpdr = state.dpdr + dpepdd;
        if constexpr (has_dpdA<T>::value) {
            state.dpdA = state.dpdA + dpepda;
        }
        if constexpr (has_dpdZ<T>::value) {
            state.dpdZ = state.dpdZ + dpepdz;
        }
    }

    if constexpr (has_entropy<T>::value) {
        state.s    = state.s + sele;
        state.dsdT = state.dsdT + dsepdt;
        state.dsdr = state.dsdr + dsepdd;
    }

    if constexpr (has_energy<T>::value) {
        state.e    = state.e + eele;
        state.dedT = state.dedT + deepdt;
        state.dedr = state.dedr + deepdd;
        if constexpr (has_dedA<T>::value) {
            state.dedA = state.dedA + deepda;
        }
        if constexpr (has_dedZ<T>::value) {
            state.dedZ = state.dedZ + deepdz;
        }
    }

    if constexpr (has_eta<T>::value) {
        state.eta = etaele;
    }

    if constexpr (has_xne_xnp<T>::value) {
        state.xne = xnefer;
        state.xnp = 0.0e0_rt;
    }

    if constexpr (has_pele_ppos<T>::value) {
        state.pele = pele;
        state.ppos = 0.0e0_rt;
    }
}



template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void apply_ions (T& state)
{
    using namespace helmholtz;

    const Real pi      = 3.1415926535897932384e0_rt;
    const Real sioncon = (2.0e0_rt * pi * amu * kerg)/(h*h);
    const Real kergavo = kerg * avo_eos;

    Real deni = 1.0e0_rt / state.rho;
    Real tempi = 1.0e0_rt / state.T;

    Real ytot1   = 1.0e0_rt / state.abar;
    Real xni     = avo_eos * ytot1 * state.rho;
    Real dxnidd  = avo_eos * ytot1;
    [[maybe_unused]] Real dxnida  = -xni * ytot1;

    Real kt = kerg * state.T;

    Real pion    = xni * kt;
    Real dpiondd = dxnidd * kt;
    Real dpiondt = xni * kerg;
    [[maybe_unused]] Real dpionda = dxnida * kt;
    [[maybe_unused]] Real dpiondz = 0.0e0_rt;

    Real eion    = 1.5e0_rt * pion * deni;
    Real deiondd = (1.5e0_rt * dpiondd - eion) * deni;
    Real deiondt = 1.5e0_rt * dpiondt * deni;
    [[maybe_unused]] Real deionda = 1.5e0_rt * dpionda * deni;
    [[maybe_unused]] Real deiondz = 0.0e0_rt;

    Real x       = state.abar * state.abar * std::sqrt(state.abar) * deni / avo_eos;
    Real s       = sioncon * state.T;
    Real z       = x * s * std::sqrt(s);
    Real y       = std::log(z);
    Real sion    = (pion * deni + eion) * tempi + kergavo * ytot1 * y;
    Real dsiondd = (dpiondd * deni - pion * deni * deni + deiondd) * tempi -
                   kergavo * deni * ytot1;
    Real dsiondt = (dpiondt * deni + deiondt) * tempi -  
                   (pion * deni + eion) * tempi * tempi +
                   1.5e0_rt * kergavo * tempi * ytot1;

    if constexpr (has_pressure<T>::value) {
        state.p    = state.p + pion;
        state.dpdT = state.dpdT + dpiondt;
        state.dpdr = state.dpdr + dpiondd;
        if constexpr (has_dpdA<T>::value) {
            state.dpdA = state.dpdA + dpionda;
        }
        if constexpr (has_dpdZ<T>::value) {
            state.dpdZ = state.dpdZ + dpiondz;
        }
    }

    if constexpr (has_energy<T>::value) {
        state.e    = state.e + eion;
        state.dedT = state.dedT + deiondt;
        state.dedr = state.dedr + deiondd;
        if constexpr (has_dedA<T>::value) {
            state.dedA = state.dedA + deionda;
        }
        if constexpr (has_dedZ<T>::value) {
            state.dedZ = state.dedZ + deiondz;
        }
    }

    if constexpr (has_entropy<T>::value) {
        state.s    = state.s + sion;
        state.dsdT = state.dsdT + dsiondt;
        state.dsdr = state.dsdr + dsiondd;
    }
}



template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void apply_radiation (T& state)
{
    using namespace helmholtz;

    const Real clight  = 2.99792458e10_rt;
#ifdef RADIATION
    const Real ssol    = 0.0e0_rt;
#else
    const Real ssol    = 5.67051e-5_rt;
#endif
    const Real asol    = 4.0e0_rt * ssol / clight;
    const Real asoli3  = asol/3.0e0_rt;

    Real deni = 1.0e0_rt / state.rho;
    Real tempi = 1.0e0_rt / state.T;

    Real prad = asoli3 * state.T * state.T * state.T * state.T;

    // In low density material, this radiation pressure becomes unphysically high.
    // For rho ~ 1 g/cc and T ~ 1e9, then this radiation pressure will lead to a
    // sound speed ~ 0.1c, and this is a problem because the codes using this EOS
    // are primarily non-relativistic. Thus we can optionally smooth out the radiation
    // pressure such that it disappears at low densities. Since all terms below depend
    // on prad, this will result in the radiation term effectively vanishing below the
    // cutoff density. For simplicity we ignore the effect this has on the derivatives.

    if (prad_limiter_rho_c > 0.0e0_rt && prad_limiter_delta_rho > 0.0e0_rt) {
        prad = prad * 0.5e0_rt * (1.0e0_rt + std::tanh((state.rho - prad_limiter_rho_c) / prad_limiter_delta_rho));
    }

    Real dpraddd = 0.0e0_rt;
    Real dpraddt = 4.0e0_rt * prad * tempi;
    [[maybe_unused]] Real dpradda = 0.0e0_rt;
    [[maybe_unused]] Real dpraddz = 0.0e0_rt;

    Real erad    = 3.0e0_rt * prad*deni;
    Real deraddd = -erad * deni;
    Real deraddt = 3.0e0_rt * dpraddt * deni;
    [[maybe_unused]] Real deradda = 0.0e0_rt;
    [[maybe_unused]] Real deraddz = 0.0e0_rt;

    Real srad    = (prad * deni + erad) * tempi;
    Real dsraddd = (dpraddd * deni - prad * deni * deni + deraddd) * tempi;
    Real dsraddt = (dpraddt * deni + deraddt - srad) * tempi;

    // Note that unlike the other terms, radiation
    // sets these terms instead of adding to them,
    // since it comes first.

    if constexpr (has_pressure<T>::value) {
        state.p    = prad;
        state.dpdr = dpraddd;
        state.dpdT = dpraddt;
        if constexpr (has_dpdA<T>::value) {
            state.dpdA = dpradda;
        }
        if constexpr (has_dpdZ<T>::value) {
            state.dpdZ = dpraddz;
        }
    }

    if constexpr (has_energy<T>::value) {
        state.e    = erad;
        state.dedr = deraddd;
        state.dedT = deraddt;
        if constexpr (has_dedA<T>::value) {
            state.dedA = deradda;
        }
        if constexpr (has_dedZ<T>::value) {
            state.dedZ = deraddz;
        }
    }

    if constexpr (has_entropy<T>::value) {
        state.s    = srad;
        state.dsdr = dsraddd;
        state.dsdT = dsraddt;
    }
}



template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void apply_coulomb_corrections (T& state)
{
    using namespace helmholtz;

    // Constants used for the Coulomb corrections
    const Real a1 = -0.898004e0_rt;
    const Real b1 =  0.96786e0_rt;
    const Real c1 =  0.220703e0_rt;
    const Real d1 = -0.86097e0_rt;
    const Real e1 =  2.5269e0_rt;
    const Real a2 =  0.29561e0_rt;
    const Real b2 =  1.9885e0_rt;
    const Real c2 =  0.288675e0_rt;
    const Real qe   = 4.8032042712e-10_rt;
    const Real esqu = qe * qe;
    const Real onethird = 1.0e0_rt/3.0e0_rt;
    const Real forth = 4.0e0_rt/3.0e0_rt;
    const Real pi    = 3.1415926535897932384e0_rt;

    [[maybe_unused]] Real pcoul    = 0.e0_rt;
    [[maybe_unused]] Real dpcouldd = 0.e0_rt;
    [[maybe_unused]] Real dpcouldt = 0.e0_rt;
    [[maybe_unused]] Real ecoul    = 0.e0_rt;
    [[maybe_unused]] Real decouldd = 0.e0_rt;
    [[maybe_unused]] Real decouldt = 0.e0_rt;
    [[maybe_unused]] Real scoul    = 0.e0_rt;
    [[maybe_unused]] Real dscouldd = 0.e0_rt;
    [[maybe_unused]] Real dscouldt = 0.e0_rt;

    [[maybe_unused]] Real dpcoulda = 0.e0_rt;
    [[maybe_unused]] Real dpcouldz = 0.e0_rt;
    [[maybe_unused]] Real decoulda = 0.e0_rt;
    [[maybe_unused]] Real decouldz = 0.e0_rt;

    Real s, x, y, z; // temporary variables

    // uniform background corrections only
    // from yakovlev & shalybkov 1989
    // lami is the average ion separation
    // plasg is the plasma coupling parameter

    Real ytot1 = 1.0e0_rt / state.abar;
    Real xni     = avo_eos * ytot1 * state.rho;
    Real dxnidd  = avo_eos * ytot1;
    [[maybe_unused]] Real dxnida  = -xni * ytot1;

    Real kt      = kerg * state.T;
    Real ktinv   = 1.0e0_rt / kt;

    z             = forth * pi;
    s             = z * xni;
    Real dsdd     = z * dxnidd;
    [[maybe_unused]] Real dsda     = z * dxnida;

    Real lami     = 1.0e0_rt / std::pow(s, onethird);
    Real inv_lami = 1.0e0_rt / lami;
    z             = -onethird * lami;
    Real lamidd   = z * dsdd / s;
    [[maybe_unused]] Real lamida   = z * dsda / s;

    Real plasg    = state.zbar * state.zbar * esqu * ktinv * inv_lami;
    z             = -plasg * inv_lami;
    Real plasgdd  = z * lamidd;
    Real plasgdt  = -plasg*ktinv * kerg;
    [[maybe_unused]] Real plasgda  = z * lamida;
    [[maybe_unused]] Real plasgdz  = 2.0e0_rt * plasg/state.zbar;

    // .yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.0e0_rt)
    {
        x        = std::pow(plasg, 0.25e0_rt);
        y        = avo_eos * ytot1 * kerg;
        ecoul    = y * state.T * (a1 * plasg + b1 * x + c1 / x + d1);
        pcoul    = onethird * state.rho * ecoul;
        scoul    = -y * (3.0e0_rt * b1 * x - 5.0e0_rt*c1 / x +
                    d1 * (std::log(plasg) - 1.0e0_rt) - e1);

        y        = avo_eos*ytot1*kt*(a1 + 0.25e0_rt/plasg*(b1*x - c1/x));
        decouldd = y * plasgdd;
        decouldt = y * plasgdt + ecoul/state.T;

        decoulda = y * plasgda - ecoul/state.abar;
        decouldz = y * plasgdz;

        y        = onethird * state.rho;
        dpcouldd = onethird * ecoul + y * decouldd;
        dpcouldt = y * decouldt;

        dpcoulda = y * decoulda;
        dpcouldz = y * decouldz;

       y        = -avo_eos * kerg / (state.abar * plasg) *
                   (0.75e0_rt * b1 * x + 1.25e0_rt * c1 / x + d1);
       dscouldd = y * plasgdd;
       dscouldt = y * plasgdt;

       // yakovlev & shalybkov 1989 equations 102, 103, 104
    }
    else if (plasg < 1.0e0_rt)
    {

        Real pion    = xni * kt;
        Real dpiondd = dxnidd * kt;
        Real dpiondt = xni * kerg;
        Real dpionda = dxnida * kt;
        Real dpiondz = 0.0e0_rt;

        x        = plasg * std::sqrt(plasg);
        y        = std::pow(plasg, b2);
        z        = c2 * x - onethird * a2 * y;
        pcoul    = -pion * z;
        ecoul    = 3.0e0_rt * pcoul / state.rho;
        scoul    = -avo_eos / state.abar * kerg * (c2 * x -a2 * (b2 - 1.0e0_rt) / b2 * y);

        s        = 1.5e0_rt * c2 * x / plasg - onethird * a2 * b2 * y / plasg;
        dpcouldd = -dpiondd * z - pion * s * plasgdd;
        dpcouldt = -dpiondt * z - pion * s * plasgdt;
        dpcoulda = -dpionda * z - pion * s * plasgda;
        dpcouldz = -dpiondz * z - pion * s * plasgdz;

        s        = 3.0e0_rt / state.rho;
        decouldd = s * dpcouldd - ecoul / state.rho;
        decouldt = s * dpcouldt;
        decoulda = s * dpcoulda;
        decouldz = s * dpcouldz;

        s        = -avo_eos * kerg / (state.abar * plasg) * 
                    (1.5e0_rt * c2 * x - a2 * (b2 - 1.0e0_rt) * y);
        dscouldd = s * plasgdd;
        dscouldt = s * plasgdt;
    }

    // Disable Coulomb corrections if they cause
    // the energy or pressure to go negative.

    Real p_temp = std::numeric_limits<Real>::max();
    Real e_temp = std::numeric_limits<Real>::max();

    if constexpr (has_pressure<T>::value) {
        p_temp = state.p + pcoul;
    }
    if constexpr (has_energy<T>::value) {
        e_temp = state.e + ecoul;
    }

    if (p_temp <= 0.0e0_rt || e_temp <= 0.0e0_rt)
    {
        pcoul    = 0.0e0_rt;
        dpcouldd = 0.0e0_rt;
        dpcouldt = 0.0e0_rt;
        ecoul    = 0.0e0_rt;
        decouldd = 0.0e0_rt;
        decouldt = 0.0e0_rt;
        scoul    = 0.0e0_rt;
        dscouldd = 0.0e0_rt;
        dscouldt = 0.0e0_rt;

        dpcoulda = 0.0e0_rt;
        dpcouldz = 0.0e0_rt;
        decoulda = 0.0e0_rt;
        decouldz = 0.0e0_rt;
    }

    if constexpr (has_pressure<T>::value) {
        state.p    = state.p + pcoul;
        state.dpdr = state.dpdr + dpcouldd;
        state.dpdT = state.dpdT + dpcouldt;
        if constexpr (has_dpdA<T>::value) {
            state.dpdA = state.dpdA + dpcoulda;
        }
        if constexpr (has_dpdZ<T>::value) {
            state.dpdZ = state.dpdZ + dpcouldz;
        }
    }

    if constexpr (has_energy<T>::value) {
        state.e    = state.e + ecoul;
        state.dedr = state.dedr + decouldd;
        state.dedT = state.dedT + decouldt;
        if constexpr (has_dedA<T>::value) {
            state.dedA = state.dedA + decoulda;
        }
        if constexpr (has_dedZ<T>::value) {
            state.dedZ = state.dedZ + decouldz;
        }
    }

    if constexpr (has_entropy<T>::value) {
        state.s    = state.s + scoul;
        state.dsdr = state.dsdr + dscouldd;
        state.dsdT = state.dsdT + dscouldt;
    }
}



template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void prepare_for_iterations (I input, T& state,
                             bool& single_iter, Real& v_want, Real& v1_want, Real& v2_want,
                             int& var, int& dvar, int& var1, int& var2)
{
    using namespace helmholtz;
    using namespace EOS;

    single_iter = true;

    if (input == eos_input_rt) {

        // Nothing to do here.

    }
    else if (input == eos_input_rh) {

        if constexpr (has_enthalpy<T>::value) {
            v_want = state.h;
            var  = ienth;
            dvar = itemp;
        }

    }
    else if (input == eos_input_tp) {

        if constexpr (has_pressure<T>::value) {
            v_want = state.p;
            var  = ipres;
            dvar = idens;
        }

    }
    else if (input == eos_input_rp) {

        if constexpr (has_pressure<T>::value) {
            v_want = state.p;
            var  = ipres;
            dvar = itemp;
        }

    }
    else if (input == eos_input_re) {

        if constexpr (has_energy<T>::value) {
            v_want = state.e;
            var  = iener;
            dvar = itemp;
        }

    }
    else if (input == eos_input_ps) {

        if constexpr (has_pressure<T>::value && has_entropy<T>::value) {
            single_iter = false;
            v1_want = state.p;
            v2_want = state.s;
            var1 = ipres;
            var2 = ientr;
        }

    }
    else if (input == eos_input_ph) {

        if constexpr (has_pressure<T>::value && has_enthalpy<T>::value) {
            single_iter = false;
            v1_want = state.p;
            v2_want = state.h;
            var1 = ipres;
            var2 = ienth;
        }

    }
    else if (input == eos_input_th) {

        if constexpr (has_enthalpy<T>::value) {
            v_want = state.h;
            var  = ienth;
            dvar = idens;
        }

    }
#ifndef AMREX_USE_GPU
    else {

        amrex::Error("Unknown EOS input");

    }
#endif
}



template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void single_iter_update (T& state, int var, int dvar,
                         Real v_want, bool& converged)
{

    using namespace helmholtz;
    using namespace EOS;

    Real x, v = 0.0_rt, dvdx = 0.0_rt, xtol, smallx;

    if (dvar == itemp) {

        x = state.T;
        smallx = EOSData::mintemp;
        xtol = ttol;

        if (var == ipres) {
            if constexpr (has_pressure<T>::value) {
                v    = state.p;
                dvdx = state.dpdT;
            }
        }
        else if (var == iener) {
            if constexpr (has_energy<T>::value) {
                v    = state.e;
                dvdx = state.dedT;
            }
        }
        else if (var == ientr) {
            if constexpr (has_entropy<T>::value) {
                v    = state.s;
                dvdx = state.dsdT;
            }
        }
        else if (var == ienth) {
            if constexpr (has_enthalpy<T>::value) {
                v    = state.h;
                dvdx = state.dhdT;
            }
        }

    }
    else // dvar == density
    {

        x = state.rho;
        smallx = EOSData::mindens;
        xtol = dtol;

        if (var == ipres) {
            if constexpr (has_pressure<T>::value) {
                v    = state.p;
                dvdx = state.dpdr;
            }
        }
        else if (var == iener) {
            if constexpr (has_energy<T>::value) {
                v    = state.e;
                dvdx = state.dedr;
            }
        }
        else if (var == ientr) {
            if constexpr (has_entropy<T>::value) {
                v    = state.s;
                dvdx = state.dsdr;
            }
        }
        else if (var == ienth) {
            if constexpr (has_enthalpy<T>::value) {
                v    = state.h;
                dvdx = state.dhdr;
            }
        }

    }

    // Now do the calculation for the next guess for T/rho
    Real xnew = x - (v - v_want) / dvdx;

    // Don't let the temperature/density change by more than a factor of two
    xnew = amrex::max(0.5_rt * x, amrex::min(xnew, 2.0_rt * x));

    // Don't let us freeze/evacuate
    xnew = amrex::max(smallx, xnew);

    // Store the new temperature/density

    if (dvar == itemp) {
        state.T = xnew;
    }
    else
    {
        state.rho = xnew;
    }

    // Compute the error from the last iteration

    Real error = std::abs((xnew - x) / x);

    if (error < xtol) converged = true;
}



template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void double_iter_update (T& state, int var1, int var2,
                         Real v1_want, Real v2_want, bool& converged)
{
    using namespace helmholtz;
    using namespace EOS;

    Real v1 = 0.0_rt, dv1dt = 0.0_rt, dv1dr = 0.0_rt, v2 = 0.0_rt, dv2dt = 0.0_rt, dv2dr = 0.0_rt;

    // Figure out which variables we're using

    Real told = state.T;
    Real rold = state.rho;

    if (var1 == ipres) {
        if constexpr (has_pressure<T>::value) {
            v1    = state.p;
            dv1dt = state.dpdT;
            dv1dr = state.dpdr;
        }
    }
    else if (var1 == iener) {
        if constexpr (has_energy<T>::value) {
            v1    = state.e;
            dv1dt = state.dedT;
            dv1dr = state.dedr;
        }
    }
    else if (var1 == ientr) {
        if constexpr (has_entropy<T>::value) {
            v1    = state.s;
            dv1dt = state.dsdT;
            dv1dr = state.dsdr;
        }
    }
    else if (var1 == ienth) {
        if constexpr (has_enthalpy<T>::value) {
            v1    = state.h;
            dv1dt = state.dhdT;
            dv1dr = state.dhdr;
        }
    }

    if (var2 == ipres) {
        if constexpr (has_pressure<T>::value) {
            v2    = state.p;
            dv2dt = state.dpdT;
            dv2dr = state.dpdr;
        }
    }
    else if (var2 == iener) {
        if constexpr (has_energy<T>::value) {
            v2    = state.e;
            dv2dt = state.dedT;
            dv2dr = state.dedr;
        }
    }
    else if (var2 == ientr) {
        if constexpr (has_entropy<T>::value) {
            v2    = state.s;
            dv2dt = state.dsdT;
            dv2dr = state.dsdr;
        }
    }
    else if (var2 == ienth) {
        if constexpr (has_enthalpy<T>::value) {
            v2    = state.h;
            dv2dt = state.dhdT;
            dv2dr = state.dhdr;
        }
    }

    // Two functions, f and g, to iterate over
    Real v1i = v1_want - v1;
    Real v2i = v2_want - v2;

    //
    // 0 = f + dfdr * delr + dfdt * delt
    // 0 = g + dgdr * delr + dgdt * delt
    //

    // note that dfi/dT = - df/dT
    Real delr = (-v1i * dv2dt + v2i * dv1dt) / (dv2dr * dv1dt - dv2dt * dv1dr);

    Real rnew = rold + delr;

    Real tnew = told + (v1i - dv1dr * delr) / dv1dt;

    // Don't let the temperature or density change by more
    // than a factor of two
    tnew = amrex::max(0.5e0_rt * told, amrex::min(tnew, 2.0e0_rt * told));
    rnew = amrex::max(0.5e0_rt * rold, amrex::min(rnew, 2.0e0_rt * rold));

    // Don't let us freeze or evacuate
    tnew = amrex::max(EOSData::mintemp, tnew);
    rnew = amrex::max(EOSData::mindens, rnew);

    // Store the new temperature and density
    state.rho = rnew;
    state.T = tnew;

    // Compute the errors
    Real error1 = std::abs((rnew - rold) / rold);
    Real error2 = std::abs((tnew - told) / told);

    if (error1 < dtol && error2 < ttol) converged = true;
}



template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void finalize_state (I input, T& state,
                     Real v_want, Real v1_want, Real v2_want)
{
    using namespace helmholtz;

    // Calculate some remaining derivatives
    if constexpr (has_pressure<T>::value) {
        state.dpde = state.dpdT / state.dedT;
        state.dpdr_e = state.dpdr - state.dpdT * state.dedr / state.dedT;
    }

    // Specific heats and Gamma_1
    if constexpr (has_energy<T>::value) {
        state.cv = state.dedT;

        if constexpr (has_pressure<T>::value) {
            Real chit = state.T / state.p * state.dpdT;
            Real chid = state.dpdr * state.rho / state.p;

            state.gam1 = (chit * (state.p / state.rho)) * (chit / (state.T * state.cv)) + chid;
            state.cp = state.cv * state.gam1 / chid;
        }
    }

    // Use the non-relativistic version of the sound speed, cs = sqrt(gam_1 * P / rho).
    // This replaces the relativistic version that comes out of helmeos.
    if constexpr (has_pressure<T>::value) {
        state.cs = std::sqrt(state.gam1 * state.p / state.rho);
    }

    if (input_is_constant) {

       if (input == eos_input_rh) {

           if constexpr (has_enthalpy<T>::value) {
               state.h = v_want;
           }

       }
       else if (input == eos_input_tp) {

           if constexpr (has_pressure<T>::value) {
               state.p = v_want;
           }

       }
       else if (input == eos_input_rp) {

           if constexpr (has_pressure<T>::value) {
               state.p = v_want;
           }

       }
       else if (input == eos_input_re) {

           if constexpr (has_energy<T>::value) {
               state.e = v_want;
           }

       }
       else if (input == eos_input_ps) {

           if constexpr (has_pressure<T>::value) {
               state.p = v1_want;
           }
           if constexpr (has_entropy<T>::value) {
               state.s = v2_want;
           }

       }
       else if (input == eos_input_ph) {

           if constexpr (has_pressure<T>::value) {
               state.p = v1_want;
           }
           if constexpr (has_enthalpy<T>::value) {
               state.h = v2_want;
           }

       }
       else if (input == eos_input_th) {

           if constexpr (has_enthalpy<T>::value) {
               state.h = v_want;
           }

       }

    }
}



template <typename I, typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void actual_eos (I input, T& state)
{
    static_assert(std::is_same<I, eos_input_t>::value, "input must be an eos_input_t");

    using namespace helmholtz;

    const int max_newton = 100;

    bool single_iter, converged;
    int var{}, dvar{}, var1, var2;
    Real v_want{}, v1_want{}, v2_want{};

    prepare_for_iterations(input, state, single_iter, v_want, v1_want, v2_want, var, dvar, var1, var2);

    converged = false;

    // Only take a single step if we're coming in with both rho and T;
    // in this call we just want the EOS to fill the other thermodynamic quantities.

    if (input == eos_input_rt) converged = true;

    // Iterate until converged.

    for (int iter = 1; iter <= max_newton; ++iter) {

        // Radiation must come first since it initializes the
        // state instead of adding to it.

        apply_radiation(state);

        apply_ions(state);

        apply_electrons(state);

        if (do_coulomb) {
            apply_coulomb_corrections(state);
        }

        // Calculate enthalpy the usual way, h = e + p / rho.

        if constexpr (has_enthalpy<T>::value) {
            state.h = state.e + state.p / state.rho;
            state.dhdr = state.dedr + state.dpdr / state.rho - state.p / (state.rho * state.rho);
            state.dhdT = state.dedT + state.dpdT / state.rho;
        }

        if (converged) {
            break;
        }
        else if (single_iter) {
            single_iter_update(state, var, dvar, v_want, converged);
        }
        else {
            double_iter_update(state, var1, var2, v1_want, v2_want, converged);
        }

    }

    finalize_state(input, state, v_want, v1_want, v2_want);
}



AMREX_INLINE
void actual_eos_init ()
{
    using namespace helmholtz;

    Real dth, dt2, dti, dt2i;
    Real dd, dd2, ddi, dd2i;
    Real tsav, dsav;

    // Read in the runtime parameters

    input_is_constant = eos_input_is_constant;
    do_coulomb = use_eos_coulomb;
    ttol = eos_ttol;
    dtol = eos_dtol;

    //    read the helmholtz free energy table
    itmax = imax;
    jtmax = jmax;
    tlo   = 3.0e0_rt;
    thi   = 13.0e0_rt;
    tstp  = (thi - tlo) / ((Real) (jmax-1));
    tstpi = 1.0e0_rt / tstp;
    dlo   = -12.0e0_rt;
    dhi   = 15.0e0_rt;
    dstp  = (dhi - dlo) / ((Real) (imax-1));
    dstpi = 1.0e0_rt / dstp;

    for (int j = 0; j < jmax; ++j) {
        tsav = tlo + j * tstp;
        t[j] = std::pow(10.0e0_rt, tsav);
        for (int i = 0; i < imax; ++i) {
            dsav = dlo + i * dstp;
            d[i] = std::pow(10.0e0_rt, dsav);
        }
    }

    // it does not work on all machines (for GPUs) broadcast to other
    // procs from managed memory.  So instead we'll read into a local
    // buffer, broadcast that, and then copy that into the managed
    // memory.

    amrex::Vector<Real> f_local(jmax * imax * 9);
    amrex::Vector<Real> dpdf_local(jmax * imax * 4);
    amrex::Vector<Real> ef_local(jmax * imax * 4);
    amrex::Vector<Real> xf_local(jmax * imax * 4);

    if (amrex::ParallelDescriptor::IOProcessor()) {

        // open the table
        std::ifstream table;
        table.open("helm_table.dat");

        if (!table.is_open()) {
            // the table was not present or we could not open it; abort
            amrex::Error("helm_table.dat could not be opened");
        }

        std::string line;

        // read in the free energy table
        int idx = 0;
        for (int j = 0; j < jmax; ++j) {
            Real T = t[j];
            for (int i = 0; i < imax; ++i) {
                Real rho = d[i];

                std::getline(table, line);
                std::istringstream data(line);
                data >> f_local[idx] >> f_local[idx+3] >> f_local[idx+1]
                     >> f_local[idx+4] >> f_local[idx+2] >> f_local[idx+5]
                     >> f_local[idx+6] >> f_local[idx+7] >> f_local[idx+8];

                eos_timmes_t state = timmes_eos(rho, T);

                Real drho = std::sqrt(std::numeric_limits<Real>::epsilon());
                eos_timmes_t state_dr = timmes_eos(rho + drho, T);

                // Fill in free energy table based on Equation 19 in Timmes and Swesty (1999)

                // f
                f_local[idx  ] = state.e - state.T * state.s;
                // df/dT
                f_local[idx+1] = -state.s;
                // d^2f/dT^2
                f_local[idx+2] = -state.dsdT;
                // df/drho
                f_local[idx+3] = state.p / (state.rho * state.rho);
                // d^2f/drho^2
                f_local[idx+4] = (state.rho * state.dpdr - 2.0_rt * state.p) / (state.rho * state.rho * state.rho);
                // d^2f/drhodT
                f_local[idx+5] = -state.dsdr;
                // d^3f/drho^2dT
                f_local[idx+6] = (state.rho * state.dpdrdT - 2.0_rt * state.dpdT) / (state.rho * state.rho * state.rho);
                // d^3f/drhodT^2
                f_local[idx+7] = -state.dsdrdT;
                // d^4f/drho^2dT^2
                f_local[idx+8] = (-state_dr.dsdrdT + state.dsdrdT) / drho;

                idx += 9;
            }
        }

        // read the pressure derivative with density table
        idx = 0;
        for (int j = 0; j < jmax; ++j) {
            for (int i = 0; i < imax; ++i) {
                std::getline(table, line);
                std::istringstream data(line);
                data >> dpdf_local[idx] >> dpdf_local[idx+2]
                     >> dpdf_local[idx+1] >> dpdf_local[idx+3];
                idx += 4;
            }
        }

        // read the electron chemical potential table
        idx = 0;
        for (int j = 0; j < jmax; ++j) {
            for (int i = 0; i < imax; ++i) {
                std::getline(table, line);
                std::istringstream data(line);
                data >> ef_local[idx] >> ef_local[idx+2]
                     >> ef_local[idx+1] >> ef_local[idx+3];
                idx += 4;
            }
        }

        // read the number density table
        idx = 0;
        for (int j = 0; j < jmax; ++j) {
            for (int i = 0; i < imax; ++i) {
                std::getline(table, line);
                std::istringstream data(line);
                data >> xf_local[idx] >> xf_local[idx+2]
                     >> xf_local[idx+1] >> xf_local[idx+3];
                idx += 4;
            }
        }

        table.close();

    }

    amrex::ParallelDescriptor::Bcast(f_local.data(),    static_cast<size_t>(9 * imax * jmax));
    amrex::ParallelDescriptor::Bcast(dpdf_local.data(), static_cast<size_t>(4 * imax * jmax));
    amrex::ParallelDescriptor::Bcast(ef_local.data(),   static_cast<size_t>(4 * imax * jmax));
    amrex::ParallelDescriptor::Bcast(xf_local.data(),   static_cast<size_t>(4 * imax * jmax));

    // now copy into managed memory
    int idx = 0;
    for (int j = 0; j < jmax; ++j) {
        for (int i = 0; i < imax; ++i) {
            for (int m = 0; m < 9; ++m) {
                f[j][i][m] = f_local[idx];
                idx++;
            }
        }
    }

    idx = 0;
    for (int j = 0; j < jmax; ++j) {
        for (int i = 0; i < imax; ++i) {
            for (int m = 0; m < 4; ++m) {
                dpdf[j][i][m] = dpdf_local[idx];
                ef[j][i][m] = ef_local[idx];
                xf[j][i][m] = xf_local[idx];
                idx++;
            }
        }
    }

    // construct the temperature and density deltas and their inverses
    for (int j = 0; j < jmax-1; ++j)
    {
        dth         = t[j+1] - t[j];
        dt2         = dth * dth;
        dti         = 1.0e0_rt / dth;
        dt2i        = 1.0e0_rt / dt2;
        dt_sav[j]   = dth;
        dt2_sav[j]  = dt2;
        dti_sav[j]  = dti;
        dt2i_sav[j] = dt2i;
    }

    for (int i = 0; i < imax-1; ++i)
    {
        dd          = d[i+1] - d[i];
        dd2         = dd * dd;
        ddi         = 1.0e0_rt / dd;
        dd2i        = 1.0e0_rt / dd2;
        dd_sav[i]   = dd;
        dd2_sav[i]  = dd2;
        ddi_sav[i]  = ddi;
        dd2i_sav[i] = dd2i;
    }

    // Set up the minimum and maximum possible densities.

    EOSData::mintemp = std::pow(10.e0_rt, tlo);
    EOSData::maxtemp = std::pow(10.e0_rt, thi);
    EOSData::mindens = std::pow(10.e0_rt, dlo);
    EOSData::maxdens = std::pow(10.e0_rt, dhi);
}



AMREX_INLINE
void actual_eos_finalize ()
{
}



template <typename I>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool is_input_valid (I input)
{
  static_assert(std::is_same<I, eos_input_t>::value, "input must be an eos_input_t");

  bool valid = true;

  return valid;
}

#endif

