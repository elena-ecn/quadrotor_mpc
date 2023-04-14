function dcm_from_mrp(p)
    p1,p2,p3 = p
    den = (p1^2 + p2^2 + p3^2 + 1)^2
    a = (4*p1^2 + 4*p2^2 + 4*p3^2 - 4)
    [
    (-((8*p2^2+8*p3^2)/den-1)*den)   (8*p1*p2 + p3*a)     (8*p1*p3 - p2*a);
    (8*p1*p2 - p3*a) (-((8*p1^2 + 8*p3^2)/den - 1)*den)   (8*p2*p3 + p1*a);
    (8*p1*p3 + p2*a)  (8*p2*p3 - p1*a)  (-((8*p1^2 + 8*p2^2)/den - 1)*den)
    ]/den
end

function vis_traj!(vis, name, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    for i = 1:(length(X)-1)
        a = X[i][1:3]
        b = X[i+1][1:3]
        cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), R)
        mc.setobject!(vis[name]["p"*string(i)], cyl, mc.MeshPhongMaterial(color=color))
    end
    for i = 1:length(X)
        a = X[i][1:3]
        sph = mc.HyperSphere(mc.Point(a...), R)
        mc.setobject!(vis[name]["s"*string(i)], sph, mc.MeshPhongMaterial(color=color))
    end
end

function animate_quadrotor(Xsim, Xref, dt)
    vis = mc.Visualizer()
    robot_obj = mc.MeshFileGeometry(joinpath(@__DIR__,"quadrotor.obj"))
    mc.setobject!(vis[:vic], robot_obj)

    vis_traj!(vis, :traj, Xref[1:85]; R = 0.01, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    target = mc.HyperSphere(mc.Point(0,0,0.0),0.1)
    mc.setobject!(vis[:target], target, mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0.0,0.4)))


    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:length(Xsim)
        mc.atframe(anim, k) do
            r = Xsim[k][1:3]
            p = Xsim[k][7:9]
            mc.settransform!(vis[:vic], mc.compose(mc.Translation(r),mc.LinearMap(1.5*(dcm_from_mrp(p)))))
            mc.settransform!(vis[:target], mc.Translation(Xref[k][1:3]))
        end
    end
    mc.setanimation!(vis, anim)

    return (mc.render(vis))
end